# Because manually writing thousands of technical neuroscience Q&A pairs 
# would take months, we will use Llama-3 (via Ollama) to autonomously 
# generate this dataset by reading chunks of the PMC neuroscience texts 
# and writing questions and answers about them.

# Step-by step pipeline to generate custom dadtaset
# 1. Ensure Ollama is running in the background of your machine. 
# Open a terminal and download the Llama-3 model:
## ollama pull llama3

import json
import os
import re
import requests
from datasets import load_dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
TARGET_PAIRS = 500           # Total number of Q&A pairs you want to collect
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"
OUTPUT_FILE = "neuro-instruction-data.json"
CHECKPOINT_EVERY = 10        # Save to disk every N successful extractions

NEURO_KEYWORDS = [
    "electroencephalography", "eeg", "brain-computer interface",
    "bci", "neural decoding", "motor imagery", "event-related potential",
    "local field potential", "spike sorting"
]

# ==========================================
# 2. FILTER & EXTRACTION LOGIC
# ==========================================
def is_neuro_paper(example):
    """Filters papers whose abstract/title contains neuro keywords."""
    try:
        front_text = " ".join(example.get("front", [])).lower()
        return any(kw in front_text for kw in NEURO_KEYWORDS)
    except Exception:
        return False

def extract_informative_paragraphs(example, min_words=60, max_words=250):
    """
    Extracts body paragraphs dense enough to contain real scientific facts,
    ignoring figure captions, single-line headers, or equations.
    """
    body_paragraphs = example.get("body", [])
    valid_chunks = []
    
    for para in body_paragraphs:
        words = para.split()
        word_count = len(words)
        
        # Keep paragraphs that are informative and mention domain terms
        if min_words <= word_count <= max_words:
            para_lower = para.lower()
            if any(kw in para_lower for kw in NEURO_KEYWORDS):
                valid_chunks.append(para.strip())
                
    return valid_chunks

# ==========================================
# 3. OLLAMA Q&A GENERATOR
# ==========================================
def prompt_llama_for_qa(text_chunk):
    prompt = f"""You are an expert computational neuroscientist. Read the following scientific passage and generate 1 to 2 high-quality, technical instruction-response pairs based strictly on the facts described.

Passage:
\"\"\"{text_chunk}\"\"\"

Format your output STRICTLY as a valid JSON array of objects with the exact keys:
- "instruction": The specific question or task.
- "input": Always leave this as an empty string "".
- "output": The direct, technically accurate answer derived from the passage.

Output valid JSON only. Do not include markdown code fences, commentary, or greetings."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": 0.1,  # Low temperature for factual precision
            "num_ctx": 2048
        },
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        response.raise_for_status()
        raw_text = response.json().get("message", {}).get("content", "").strip()

        # Clean potential markdown fences (e.g., ```json ... ```)
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.MULTILINE).strip()
        
        data = json.loads(cleaned)
        if isinstance(data, list):
            # Validate required fields
            valid_pairs = []
            for item in data:
                if isinstance(item, dict) and "instruction" in item and "output" in item:
                    valid_pairs.append({
                        "instruction": item["instruction"].strip(),
                        "input": item.get("input", "").strip(),
                        "output": item["output"].strip()
                    })
            return valid_pairs
    except (requests.RequestException, json.JSONDecodeError):
        return []

    return []

# ==========================================
# 4. MAIN STREAMING & ACCUMULATION PIPELINE
# ==========================================
def main():
    # Load existing progress if available
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Resuming: Loaded {len(dataset)} existing Q&A pairs from {OUTPUT_FILE}.")
    else:
        dataset = []

    print("Connecting to Hugging Face PMC full-text stream...")
    raw_stream = load_dataset(
        "TomTBT/pmc_open_access_xml",
        "commercial",
        split="train",
        streaming=True
    )
    
    neuro_stream = raw_stream.filter(is_neuro_paper)

    print(f"Harvesting Q&A pairs until reaching {TARGET_PAIRS} entries...")

    pairs_since_last_save = 0

    for paper in neuro_stream:
        if len(dataset) >= TARGET_PAIRS:
            break

        paragraphs = extract_informative_paragraphs(paper)
        if not paragraphs:
            continue

        for chunk in paragraphs:
            if len(dataset) >= TARGET_PAIRS:
                break

            qa_pairs = prompt_llama_for_qa(chunk)
            if qa_pairs:
                dataset.extend(qa_pairs)
                pairs_since_last_save += len(qa_pairs)
                print(f"[Collected {len(dataset)}/{TARGET_PAIRS}] Added {len(qa_pairs)} pair(s) from chunk.")

            # Periodic checkpoint saving
            if pairs_since_last_save >= CHECKPOINT_EVERY:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=4)
                pairs_since_last_save = 0
                print(f"--- Saved checkpoint: {len(dataset)} pairs written to {OUTPUT_FILE} ---")

    # Final save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"\nCompleted! Successfully compiled {len(dataset)} instruction pairs into {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()