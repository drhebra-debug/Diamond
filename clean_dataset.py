import zipfile
import csv
import re
import os
from pathlib import Path
from langdetect import detect, LangDetectException

# =========================
# CONFIG
# =========================

ZIP_PATH = "dataset.zip"  # change if needed
OUTPUT_FILE = "cleaned_dataset.csv"
MIN_WORDS = 5

# =========================
# UTILS
# =========================

def contains_html(text: str) -> bool:
    html_patterns = [
        r"<[^>]+>",
        r"&nbsp;",
        r"&amp;",
        r"&lt;",
        r"&gt;",
        r"style=",
        r"class=",
        r"script",
        r"div",
        r"http[s]?://"
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in html_patterns)


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def has_code(text: str) -> bool:
    code_patterns = [
        r"def ",
        r"class ",
        r"import ",
        r"from ",
        r"SELECT ",
        r"INSERT ",
        r"CREATE TABLE",
        r"for ",
        r"while ",
        r"\{.*\}",
        r"if __name__",
        r"console\.log",
        r"#!/",
        r"pip install",
        r"docker ",
        r"kubectl ",
        r"apt-get",
        r"sudo ",
        r"\.py",
        r"\.ts",
        r"\.js"
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in code_patterns)


def meaningful_word_count(text: str) -> int:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    return len(words)


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# MAIN CLEANING LOGIC
# =========================

def extract_csv_from_zip(zip_path: str) -> Path:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("extracted")
    for root, _, files in os.walk("extracted"):
        for file in files:
            if file.endswith(".csv"):
                return Path(root) / file
    raise FileNotFoundError("No CSV file found in ZIP.")


def clean_csv(input_csv: Path, output_csv: str):
    total = 0
    kept = 0

    with open(input_csv, "r", encoding="utf-8", errors="ignore") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader, None)
        if header:
            writer.writerow(header)

        for row in reader:
            total += 1

            if not row:
                continue

            text = " ".join(row).strip()

            # Remove empty
            if not text:
                continue

            # Remove short
            if meaningful_word_count(text) < MIN_WORDS:
                continue

            # Remove HTML
            if contains_html(text):
                continue

            # English only
            if not is_english(text):
                continue

            # Must contain code
            if not has_code(text):
                continue

            cleaned = clean_text(text)
            writer.writerow([cleaned])
            kept += 1

    print(f"Total rows scanned: {total}")
    print(f"Rows kept: {kept}")
    print(f"Rows removed: {total - kept}")
    print(f"Cleaned dataset saved to: {output_csv}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    csv_path = extract_csv_from_zip(ZIP_PATH)
    print(f"Found CSV: {csv_path}")
    clean_csv(csv_path, OUTPUT_FILE)
