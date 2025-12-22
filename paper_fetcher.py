import os
import time
import requests
import json
import logging
from pathlib import Path

# --- 1. The "Golden Set" with DIRECT IDs (No Searching Needed) ---
# We use ArXiv IDs where possible as they are resolvable by S2 API.
SEED_PAPERS = {
    "Attention Is All You Need": "ARXIV:1706.03762",
    "BERT": "ARXIV:1810.04805",
    "Deep Residual Learning for Image Recognition": "ARXIV:1512.03385",
    "Adam: A Method for Stochastic Optimization": "ARXIV:1412.6980",
    "Layer Normalization": "ARXIV:1607.06450",
    "Sequence to Sequence Learning with Neural Networks": "ARXIV:1409.3215",
    "Neural Machine Translation (Bahdanau Attention)": "ARXIV:1409.0473",
    "Playing Atari with Deep Reinforcement Learning": "ARXIV:1312.5602",
    "Generative Adversarial Networks": "ARXIV:1406.2661",
    "Dropout": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",  # S2 ID (Dropout sometimes tricky on ArXiv versions)
    "Batch Normalization": "ARXIV:1502.03167",
    "Mask R-CNN": "ARXIV:1703.06870",
    "U-Net": "ARXIV:1505.04597",
    "You Only Look Once (YOLO)": "ARXIV:1506.02640",
    "Distilling the Knowledge in a Neural Network": "ARXIV:1503.02531",
}

# --- Configuration ---
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"


# ArXiv requires a polite User-Agent
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Accept": "application/pdf",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def setup_directories():
    os.makedirs("seed_papers", exist_ok=True)


def get_credentials():
    key_filename = "api_key.json"
    file_path = Path(key_filename)
    if file_path.exists():
        with open(file_path, "r") as file:
            # Deserialize the file content into a Python dictionary
            headers = json.load(file)
            logger.info(f"Found {key_filename}")
    else:
        logger.info(f"Didn't find {key_filename}, setting empty key")
        headers = {}
    return headers


def robust_api_call(url, params=None, retries=4, headers={}):
    """
    Makes an API call with exponential backoff for 429/5xx errors.
    """

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 200:
                return response.json()

            # Rate Limit (429) or Server Error (500/504) -> Wait and Retry
            if response.status_code in [429, 500, 502, 503, 504]:
                wait_time = (2**attempt) * 5  # 5s, 10s, 20s, 40s
                logger.warning(
                    f"    !! Status {response.status_code}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue

            # Client Error (400/404) -> Do not retry
            logger.error(f"    !! Client Error {response.status_code} for {url}")
            return None

        except requests.exceptions.RequestException as e:
            wait_time = (2**attempt) * 5
            logger.warning(f"    !! Network Error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    logger.error("    !! Max retries reached. Giving up.")
    return None


def get_paper_details(paper_id, headers):
    """Fetches details using the ID directly (Bypasses Search)."""
    url = f"{S2_BASE_URL}/paper/{paper_id}"
    params = {"fields": "paperId,title,corpusId,openAccessPdf,externalIds"}
    return robust_api_call(url, params, headers=headers)


def get_best_pdf_url(paper_data):
    # 1. Try manual ArXiv construction (Most reliable)
    if "externalIds" in paper_data and "ArXiv" in paper_data["externalIds"]:
        arxiv_id = paper_data["externalIds"]["ArXiv"]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    # 2. Try the API's provided OpenAccess URL
    if paper_data.get("openAccessPdf") and paper_data["openAccessPdf"].get("url"):
        return paper_data["openAccessPdf"]["url"]

    return None


def download_pdf(url, filename):
    """Downloads PDF with backoff."""
    if not url:
        return False

    safe_name = "".join(
        [c for c in filename if c.isalnum() or c in (" ", "-", "_")]
    ).rstrip()
    filepath = os.path.join("seed_papers", f"{safe_name}.pdf")

    if os.path.exists(filepath):
        logger.info(f"    [Exists] {safe_name}")
        return True

    for attempt in range(3):
        try:
            time.sleep(3)  # Polite wait
            response = requests.get(
                url, headers=DOWNLOAD_HEADERS, stream=True, timeout=30
            )

            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"    [Downloaded] {safe_name}")
                return True
            elif response.status_code == 429:
                wait = (attempt + 1) * 10
                logger.warning(f"    [Busy] ArXiv 429. Waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.warning(f"    [Failed] Status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"    [Error] {e}")

    return False


def get_citations(paper_id, headers):
    """Fetches citation metadata (IDs only)."""
    url = f"{S2_BASE_URL}/paper/{paper_id}/references"
    params = {
        "fields": "paperId,title,corpusId",
        "limit": 500,  # Cap to prevent timeouts on huge papers
    }
    data = robust_api_call(url, params, headers=headers)
    if not data:
        return []

    refs = []
    for r in data.get("data", []):
        if r.get("citedPaper") and r["citedPaper"].get("paperId"):
            refs.append(r["citedPaper"])
    return refs


def fetch_papers():
    setup_directories()
    master_map = {}

    for title, pid in SEED_PAPERS.items():
        logger.info(f"--- Processing: {title} ({pid}) ---")

        headers = get_credentials()

        # 1. Get Details (Direct ID Lookup)
        paper_data = get_paper_details(pid, headers=headers)
        if not paper_data:
            logger.error(f"Could not retrieve metadata for {title}")
            continue

        real_title = paper_data["title"]
        real_id = paper_data["paperId"]

        # 2. Download PDF
        pdf_url = get_best_pdf_url(paper_data)
        if not download_pdf(pdf_url, real_title):
            logger.warning(f"    [Skipping Download] No valid PDF URL found.")
            # We continue anyway to get the citations for the map

        # 3. Get Citations
        logger.info("    Fetching citations...")
        refs = get_citations(real_id, headers=headers)

        citations_map = {}
        for ref in refs:
            ref_title = ref["title"]
            if ref_title:
                citations_map[ref_title] = {
                    "paperId": ref["paperId"],
                    "corpusId": ref.get("corpusId"),
                }

        # 4. Save to Master Map
        master_map[real_title] = {
            "seed_id": real_id,
            "citations_by_title": citations_map,
        }

        with open("citation_resolver_map.json", "w") as f:
            json.dump(master_map, f, indent=4)

        logger.info(f"    Mapped {len(citations_map)} citations.")
        time.sleep(2)
