import os
import subprocess
import shutil
import torch  # Need torch to detect the device


def ocr_directory_marker(input_dir, output_dir):
    """
    Uses 'marker-pdf' to convert PDFs to Markdown.
    Auto-detects CUDA (Colab) vs MPS (Mac).
    """
    os.makedirs(output_dir, exist_ok=True)

    temp_marker_dir = os.path.join(output_dir, "temp_marker_raw")
    os.makedirs(temp_marker_dir, exist_ok=True)

    results = {}

    # --- 🔍 DEVICE DETECTION START ---
    my_env = os.environ.copy()

    if torch.cuda.is_available():
        print("⚡️ Device Detected: NVIDIA GPU (CUDA)")
        my_env["TORCH_DEVICE"] = "cuda"
    elif torch.backends.mps.is_available():
        print("🍎 Device Detected: Apple Silicon (MPS)")
        my_env["TORCH_DEVICE"] = "mps"
    else:
        print("🐢 Device Detected: CPU (Slow)")
        my_env["TORCH_DEVICE"] = "cpu"
    # --- DEVICE DETECTION END ---

    print(f"🚀 Starting Marker OCR on '{input_dir}'...")

    if os.path.exists(input_dir):
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(input_dir, filename)
                file_stem = filename.rsplit(".", 1)[0]

                clean_output_path = os.path.join(output_dir, f"{file_stem}.mmd")

                # Skip Logic
                if os.path.exists(clean_output_path):
                    print(f"⏭️  Skipping: {filename} (Already exists)")
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

                        print(f"  ✅ Success")
                    else:
                        print(f"  ❌ Error: Marker finished but output file missing.")

                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Crash: Marker failed for {filename}")
                    # On Colab, print the error so you can see if dependencies are missing
                    print(e.stderr.decode())

    return results


if __name__ == "__main__":
    paper_contents = ocr_directory_marker("seed_papers", "seed_papers_md")
    print(f"\nDone! Processed {len(paper_contents)} papers.")
