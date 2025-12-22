import os
import subprocess
import shutil


def ocr_directory_marker(input_dir, output_dir):
    """
    Uses 'marker-pdf' to convert PDFs to Markdown.
    Skips processing if the .mmd file already exists in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    temp_marker_dir = os.path.join(output_dir, "temp_marker_raw")
    os.makedirs(temp_marker_dir, exist_ok=True)

    results = {}

    print(f"🚀 Starting Marker OCR on '{input_dir}'...")

    if os.path.exists(input_dir):
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(input_dir, filename)
                file_stem = filename.rsplit(".", 1)[0]

                # Define the final destination path ahead of time
                clean_output_path = os.path.join(output_dir, f"{file_stem}.mmd")

                # --- ⚡️ SKIP LOGIC START ⚡️ ---
                if os.path.exists(clean_output_path):
                    print(f"⏭️  Skipping: {filename} (Already exists)")

                    # We still read the file so your downstream code gets the data
                    with open(clean_output_path, "r", encoding="utf-8") as f:
                        results[filename] = f.read()
                    continue  # Jump to the next file
                # --- SKIP LOGIC END ---

                print(f"Processing: {filename}")

                # UPDATED COMMAND (Marker v1.0+ style)
                command = ["marker_single", pdf_path, "--output_dir", temp_marker_dir]

                # OPTIONAL: Force GPU on Mac (remove 'env=my_env' below if not needed)
                my_env = os.environ.copy()
                my_env["TORCH_DEVICE"] = "mps"

                try:
                    subprocess.run(command, check=True, capture_output=True, env=my_env)

                    # Locate the generated markdown file
                    # Marker structure: temp_dir / file_stem / file_stem.md
                    expected_md_path = os.path.join(
                        temp_marker_dir, file_stem, f"{file_stem}.md"
                    )

                    if os.path.exists(expected_md_path):
                        # Read the content
                        with open(expected_md_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Save to results
                        results[filename] = content

                        # Save clean copy to main folder (this enables the skip logic next time)
                        with open(clean_output_path, "w", encoding="utf-8") as f:
                            f.write(content)

                        print(f"  ✅ Success")
                    else:
                        print(f"  ❌ Error: Marker finished but output file missing.")

                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Crash: Marker failed for {filename}")
                    print(e.stderr.decode())

    return results


if __name__ == "__main__":
    paper_contents = ocr_directory_marker("seed_papers", "seed_papers_md")
    print(f"\nDone! Processed {len(paper_contents)} papers.")
