import os
import re
import json
import unicodedata
from glob import glob
from tqdm import tqdm

HEADER_FOOTER_PATTERNS = [
    r"^\s*page\s*\d+\s*(of\s*\d+)?\s*$", r"^\s*\d+\s*$", r"^\s*(copyright|©)\b.*$",
    r"^\s*confidential\b.*$", r"^\s*for internal use only\b.*$",
]
BULLET_MARKS = [r"•", r"▪", r"‣", r"–", r"—", r"·", r"\*", r"∙", r"●", r"◦"]

def _looks_like_noise(s: str) -> bool:
    if not s.strip(): return True
    letters = sum(ch.isalpha() for ch in s)
    if letters / max(1, len(s)) < 0.4: return True
    if re.search(r"(.)\1{4,}", s): return True
    return False

def _normalize_bullets(s: str) -> str:
    bullet_union = "|".join(BULLET_MARKS)
    return re.sub(rf"^\s*(?:{bullet_union})\s*", "- ", s, flags=re.MULTILINE)

def clean_text(raw: str) -> str:
    if not raw: return ""
    s = unicodedata.normalize("NFKC", raw)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    cleaned_lines = []
    for ln in lines:
        stripped_ln = ln.strip()
        if any(re.match(p, stripped_ln, flags=re.IGNORECASE) for p in HEADER_FOOTER_PATTERNS): continue
        if _looks_like_noise(stripped_ln): continue
        cleaned_lines.append(ln)
    s = "\n".join(cleaned_lines)
    s = _normalize_bullets(s).strip()
    if len(s) < 50: return ""
    return s

def main():
    INPUT_DIR = "data_chunks"
    OUTPUT_DIR = "data_clean_chunks"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = glob(os.path.join(INPUT_DIR, "*.jsonl"))
    if not files:
        print(f"No JSONL files found in {INPUT_DIR}. Run process_pdfs.py first.")
        return
    print(f"🧾 Found {len(files)} JSONL files to clean.")
    for file_path in tqdm(files, desc="Cleaning JSONL files"):
        filename = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        cleaned_count, original_count = 0, 0
        with open(file_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                original_count += 1
                try:
                    obj = json.loads(line)
                    text = clean_text(obj.get("text", ""))
                    if not text: continue
                    obj["text"] = text
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    cleaned_count += 1
                except json.JSONDecodeError: continue
        print(f"✅ {filename}: Kept {cleaned_count} of {original_count} chunks.")
    print("🎉 All files cleaned successfully.")

if __name__ == "__main__":
    main()