import os
import subprocess
import torch
import re
import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from thefuzz import process  # For fuzzy matching titles
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

CITATION_MAP_PATH = "citation_resolver_map.json"
DB_PATH = "paper_rag_db"
EMBED_MODEL_NAME = "all-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def ocr_directory_marker(input_dir, output_dir):
    """
    Uses 'marker-pdf' to convert PDFs to Markdown.
    """
    os.makedirs(output_dir, exist_ok=True)

    temp_marker_dir = os.path.join(output_dir, "temp_marker_raw")
    os.makedirs(temp_marker_dir, exist_ok=True)

    results = {}

    # Device detection logic
    my_env = os.environ.copy()

    if torch.cuda.is_available():
        print("Device Detected: NVIDIA GPU (CUDA)")
        my_env["TORCH_DEVICE"] = "cuda"
    elif torch.backends.mps.is_available():
        print("Device Detected: Apple Silicon (MPS)")
        my_env["TORCH_DEVICE"] = "mps"
    else:
        print("Device Detected: CPU (Slow)")
        my_env["TORCH_DEVICE"] = "cpu"

    print(f"Starting Marker OCR on '{input_dir}'...")

    if os.path.exists(input_dir):
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(input_dir, filename)
                file_stem = filename.rsplit(".", 1)[0]

                clean_output_path = os.path.join(output_dir, f"{file_stem}.mmd")

                # Skip Logic
                if os.path.exists(clean_output_path):
                    print(f"Skipping: {filename} (Already exists)")
                    with open(clean_output_path, "r", encoding="utf-8") as f:
                        results[filename] = f.read()
                    continue

                print(f"Processing: {filename}")

                command = ["marker_single", pdf_path, "--output_dir", temp_marker_dir]

                try:
                    # Pass the env with the correct device to the subprocess
                    subprocess.run(command, check=True, capture_output=True, env=my_env)

                    expected_md_path = os.path.join(
                        temp_marker_dir, file_stem, f"{file_stem}.md"
                    )

                    if os.path.exists(expected_md_path):
                        with open(expected_md_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        results[filename] = content

                        with open(clean_output_path, "w", encoding="utf-8") as f:
                            f.write(content)

                        print(f"Success")
                    else:
                        print(f"Error: Marker finished but output file missing.")

                except subprocess.CalledProcessError as e:
                    print(f"Crash: Marker failed for {filename}")
                    # On Colab, print the error so you can see if dependencies are missing
                    print(e.stderr.decode())

    return results


def run_pipeline(input_dir):
    # Initialize Models
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    enricher = ContextEnricher()
    resolver = CitationResolver(CITATION_MAP_PATH)

    # Initialize DB
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name="scientific_papers")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    print(f"Scanning {input_dir}...")

    files = [
        f for f in os.listdir(input_dir) if f.endswith(".mmd") or f.endswith(".md")
    ]

    for filename in files:
        print(f"Processing {filename}...")
        file_path = os.path.join(input_dir, filename)

        # Use filename (minus extension) as a proxy for the paper title key
        # Ideally, you'd extract the real title from the first lines of the MD
        paper_title_key = filename.rsplit(".", 1)[0]

        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Step A: Parse Bibliography first (for Local Resolution)
        local_bib = resolver.parse_local_bibliography(full_text)

        # Step B: Split Text
        chunks = splitter.create_documents([full_text])

        # Buffers for batch adding
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content

            # Step C: Resolve Citations
            ref_ids, ref_corpus_ids = resolver.resolve(
                chunk_text, local_bib, paper_title_key
            )

            # Step D: Enrich (Contextual Retrieval)
            # We generate a summary "header" and prepend it
            context_header = enricher.generate_context(chunk_text, paper_title_key)
            expanded_text = f"{context_header}\n\n{chunk_text}"

            # Step E: Embed the EXPANDED text
            vector = embedder.encode(expanded_text).tolist()

            # Prepare for DB
            chunk_id = f"{paper_title_key}_chunk_{i}"

            ids.append(chunk_id)
            documents.append(chunk_text)  # Store ORIGINAL text for display
            embeddings.append(vector)  # Search via EXPANDED vector
            metadatas.append(
                {
                    "source": paper_title_key,
                    "enriched_context": context_header,  # Save this to see what LLM added
                    "citation_ids": str(
                        ref_ids
                    ),  # Storing lists as strings (Chroma limitation)
                    "citation_corpus_ids": str(ref_corpus_ids),
                }
            )

            if i % 5 == 0:
                print(f"  Processed chunk {i}/{len(chunks)}...")

        # Step F: Store Batch
        if ids:
            collection.add(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )
            print(f"  Saved {len(ids)} chunks to ChromaDB.")


class ContextEnricher:
    def __init__(self):
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        print(f"Loading LLM for Enrichment ({model_id})...")
        # Load in 4-bit to fit on Colab T4 GPU
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config, device_map="auto"
        )

    def generate_context(self, chunk_text: str, paper_title: str) -> str:
        """
        Generates a concise context string for the chunk.
        """
        # We construct a standard message list
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant helping to index scientific papers.",
            },
            {
                "role": "user",
                "content": (
                    f"The following text is a chunk from the paper '{paper_title}'.\n"
                    "Please write a short, one-sentence explanation of what this chunk discusses to provide context for retrieval.\n"
                    "Do not mention 'this chunk' or 'the text', just describe the content directly.\n\n"
                    f'Chunk: "{chunk_text[:1000]}..."'
                ),
            },
        ]

        # AUTO-FORMATTING: This handles the specific "<|im_start|>" or "<|begin_of_text|>" for you
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=50, do_sample=True, temperature=0.3
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Robustly extract only the assistant's new response
        # Qwen and others usually repeat the prompt in 'generated_text', so we split it out.
        # This split logic works for most chat templates which simply append the answer.
        if "assistant\n" in generated_text:
            # Llama style
            response = generated_text.split("assistant\n")[-1].strip()
        elif "assistant" in generated_text:
            # Qwen/Generic style often looks like "...assistant\nResponse..."
            response = generated_text.split("assistant")[-1].strip()
        else:
            # Fallback if the template is weird, just return the whole tail
            # (You might need to adjust this depending on exact raw output)
            response = generated_text[len(text_input) :].strip()

        return response


class CitationResolver:
    def __init__(self, map_path):
        with open(map_path, "r") as f:
            self.global_map = json.load(f)

    def parse_local_bibliography(self, full_text):
        """
        Scans the end of the markdown to find lines like '[1] Vaswani et al. "Attention..."'
        Returns a dict: { "1": "Attention is all you need..." }
        """
        full_text_lower = full_text.lower()
        # Marker usually puts references at the end. We look for the References header.
        # This regex looks for [number] followed by text
        refs = {}
        # Simple heuristic: Split by "References" or "Bibliography"
        if "references" in full_text_lower:
            bib_section = full_text.split("References")[-1]
        elif "bibliography" in full_text_lower:
            bib_section = full_text.split("Bibliography")[-1]
        else:
            return {}  # Could not find bibliography

        # Regex to find [1] Title...
        pattern = re.compile(r"\[(\d+)\]\s*(.+)")
        for line in bib_section.split("\n"):
            match = pattern.search(line)
            if match:
                ref_num = match.group(1)
                ref_text = match.group(2)
                refs[ref_num] = ref_text
        return refs

    def resolve(self, chunk_text, local_bib, paper_title_key):
        """
        1. Finds [x] in chunk.
        2. Looks up [x] in local_bib to get Title.
        3. Fuzzy matches Title in global_map to get ID.
        """
        found_ids = []
        found_corpus_ids = []

        # 1. Find citations in text like [1], [12], etc.
        citations_in_chunk = re.findall(r"\[(\d+)\]", chunk_text)

        if not citations_in_chunk or not local_bib:
            return [], []

        # Get the specific map for this seed paper
        # We need to find the matching key in the JSON (handling potential minor title differences)
        best_paper_match, score = process.extractOne(
            paper_title_key, self.global_map.keys()
        )
        if score < 90:
            print(
                f"Warning: Could not find seed paper '{paper_title_key}' in JSON map."
            )
            return [], []

        seed_data = self.global_map[best_paper_match]["citations_by_title"]
        valid_titles = list(seed_data.keys())

        for ref_num in citations_in_chunk:
            # 2. Local Lookup (Number -> Title)
            ref_string = local_bib.get(ref_num)
            if not ref_string:
                continue

            # 3. Global Lookup (Title -> ID) via Fuzzy Match
            # We match the text from the PDF bibliography against the clean API titles
            match_title, score = process.extractOne(ref_string, valid_titles)

            if score > 70:  # Threshold for fuzzy match
                matched_data = seed_data[match_title]
                found_ids.append(matched_data.get("paperId"))
                if matched_data.get("corpusId"):
                    found_corpus_ids.append(matched_data.get("corpusId"))

        return list(set(found_ids)), list(set(found_corpus_ids))


if __name__ == "__main__":
    ocr_output_dir = "seed_papers_md"
    paper_contents = ocr_directory_marker("seed_papers", ocr_output_dir)
    run_pipeline(ocr_output_dir)
    print(f"\nDone! Processed {len(paper_contents)} papers.")
