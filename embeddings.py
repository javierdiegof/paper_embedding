import os
import subprocess
import shutil


def ocr_directory_marker(input_dir, output_dir):
    """
    Uses 'marker-pdf' to convert PDFs to Markdown.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Marker creates a lot of temporary files/images.
    # We will use this temp folder to store the raw marker output.
    temp_marker_dir = os.path.join(output_dir, "temp_marker_raw")
    os.makedirs(temp_marker_dir, exist_ok=True)

    results = {}

    print(f"🚀 Starting Marker OCR on '{input_dir}'...")

    if os.path.exists(input_dir):
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(input_dir, filename)
                file_stem = filename.rsplit(".", 1)[0]

                # Define the specific output folder for this file (Marker requirement)
                # Marker will output to: temp_marker_dir/file_stem/file_stem.md

                print(f"Processing: {filename}")

                # Construct the command
                # --batch_multiplier 2 helps saturate the M1 GPU/CPU
                # --langs English ensures it focuses on English text
                command = ["marker_single", pdf_path, "--output_dir", temp_marker_dir]

                try:
                    # Run Marker as a subprocess
                    # capture_output=True keeps your terminal clean from Marker's verbose logs
                    subprocess.run(command, check=True, capture_output=True)

                    # Locate the generated markdown file
                    # Marker structure: output_folder / pdf_name / pdf_name.md
                    expected_md_path = os.path.join(
                        temp_marker_dir, file_stem, f"{file_stem}.md"
                    )

                    if os.path.exists(expected_md_path):
                        # Read the content
                        with open(expected_md_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Save it to your results dict
                        results[filename] = content

                        # OPTIONAL: Save a clean copy to your main output folder (flattened)
                        clean_output_path = os.path.join(output_dir, f"{file_stem}.mmd")
                        with open(clean_output_path, "w", encoding="utf-8") as f:
                            f.write(content)

                        print(f"  ✅ Success")
                    else:
                        print(f"  ❌ Error: Marker finished but output file missing.")

                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Crash: Marker failed for {filename}")
                    print(e.stderr.decode())  # Uncomment to debug specific crashes

    # Cleanup: Remove the folder with extracted images/JSONs if you don't need them
    # shutil.rmtree(temp_marker_dir)

    return results


if __name__ == "__main__":
    paper_contents = ocr_directory_marker("seed_papers", "seed_papers_md")
    print(f"\nDone! Processed {len(paper_contents)} papers.")
