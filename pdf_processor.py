import os
import re
import json
import uuid
import time
import argparse
import pathlib
import io
from typing import List, Tuple
import pdfplumber
from tqdm import tqdm
from PIL import Image

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# --- Poppler Path for Windows ---
POPPLER_PATH = r"C:\poppler\poppler-25.07.0\Library\bin"

# --- Image extraction settings ---
MIN_IMAGE_SIZE = 100  # Minimum width/height in pixels to process
MIN_IMAGE_AREA = 10000  # Minimum area (width * height) to process


def extract_images_from_page(page) -> List[Image.Image]:
    """Extract embedded images from a PDF page using pdfplumber."""
    images = []
    try:
        for img_info in page.images:
            # Get image bounding box
            x0, y0, x1, y1 = img_info['x0'], img_info['top'], img_info['x1'], img_info['bottom']
            width = x1 - x0
            height = y1 - y0
            
            # Skip small images (likely icons, bullets, etc.)
            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                continue
            if width * height < MIN_IMAGE_AREA:
                continue
            
            # Crop the image region from the page
            cropped = page.within_bbox((x0, y0, x1, y1))
            img = cropped.to_image(resolution=150)
            
            # Convert to PIL Image
            pil_img = img.original
            images.append(pil_img)
    except Exception as e:
        pass  # Silently skip if image extraction fails
    
    return images


def ocr_image(img: Image.Image) -> str:
    """Run OCR on a PIL Image and return extracted text."""
    if not OCR_AVAILABLE:
        return ""
    try:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        text = pytesseract.image_to_string(img, lang='eng')
        return text.strip()
    except Exception:
        return ""


def read_pdf_text_pages(pdf_path: str, dpi: int = 300, ocr_threshold: int = 60, extract_images: bool = True) -> List[str]:
    """
    Extract text from PDF pages with enhanced image support.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for full-page OCR
        ocr_threshold: If extracted text is below this, do full-page OCR
        extract_images: If True, also extract and OCR embedded images
    """
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(tqdm(pdf.pages, desc=f"Reading {os.path.basename(pdf_path)}")):
            page_texts = []
            
            # 1. Extract regular text
            txt = (page.extract_text() or '').strip()
            page_texts.append(txt)
            
            # 2. Check if page has embedded images and extract text from them
            has_images = len(page.images) > 0 if hasattr(page, 'images') else False
            
            if extract_images and has_images and OCR_AVAILABLE:
                try:
                    # Extract and OCR embedded images
                    embedded_imgs = extract_images_from_page(page)
                    for img in embedded_imgs:
                        img_text = ocr_image(img)
                        if img_text and len(img_text) > 20:  # Only add meaningful text
                            page_texts.append(f"[Image Text]: {img_text}")
                except Exception as e:
                    print(f"Warning: Image extraction failed on page {i+1}: {e}")
            
            # 3. If very little text extracted, do full-page OCR
            combined_text = "\n".join(page_texts)
            if len(combined_text.replace("[Image Text]:", "").strip()) < ocr_threshold and OCR_AVAILABLE:
                try:
                    page_images = convert_from_path(
                        pdf_path, 
                        first_page=i + 1, 
                        last_page=i + 1, 
                        dpi=dpi,
                        poppler_path=POPPLER_PATH
                    )
                    if page_images:
                        ocr_text = (pytesseract.image_to_string(page_images[0]) or '').strip()
                        if ocr_text:
                            page_texts = [ocr_text]  # Replace with full OCR
                except Exception as e:
                    print(f"Warning: Full-page OCR failed on page {i+1} of {pdf_path}: {e}")
            
            texts.append("\n".join(page_texts))
    return texts

def clean_and_stitch_text(pages: List[str]) -> str:
    full_text = "\n\n".join(pages)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r"[ \t]{2,}", " ", full_text)
    full_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", full_text)
    return full_text.strip()

def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text): break
        start += chunk_size - chunk_overlap
    return [c.strip() for c in chunks if c.strip()]

def process_pdf(pdf_path: str, out_dir: str, chunk_size: int, chunk_overlap: int, ocr_threshold: int, extract_images: bool = True) -> str:
    """Process a PDF file into chunked JSONL with image extraction support."""
    pages = read_pdf_text_pages(pdf_path, ocr_threshold=ocr_threshold, extract_images=extract_images)
    if not pages: raise RuntimeError(f"No text extracted from {pdf_path}")
    
    full_text = clean_and_stitch_text(pages)
    if not full_text: raise RuntimeError(f"No text remained after cleaning {pdf_path}")
    
    chunks = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_id = str(uuid.uuid4())
    title = pathlib.Path(pdf_path).stem.replace("_", " ").title()
    out_path = os.path.join(out_dir, f"{pathlib.Path(pdf_path).stem}.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        
        # --- FIX 2: Rename the local variable ---
        # We change 'chunk_text' to 'chunk_content' to avoid the bug
        for i, chunk_content in enumerate(chunks):
            record = {
                "id": str(uuid.uuid4()), "doc_id": doc_id, "title": title,
                "source_file": str(pdf_path), "chunk_index": i, "text": chunk_content,
                "created_at": time.strftime("%Y-%m-%d")
            }
            # ----------------------------------------
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Process PDFs into chunked JSONL files with image OCR support.")
    parser.add_argument("--input_dir", default="pdfs", help="Input folder with PDFs.")
    parser.add_argument("--output_dir", default="./data_chunks", help="Output folder for JSONL files.")
    parser.add_argument("--chunk_size", type=int, default=1500)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--ocr_threshold", type=int, default=60)
    parser.add_argument("--no_images", action="store_true", help="Disable image extraction (faster)")
    args = parser.parse_args()
    
    extract_images = not args.no_images
    
    pdf_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(".pdf")]
    if not pdf_paths:
        print(f"No PDFs found in: {args.input_dir}")
        return
        
    print(f"Found {len(pdf_paths)} PDFs. OCR available: {OCR_AVAILABLE}. Image extraction: {extract_images}")
    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        try:
            out_path = process_pdf(pdf_path, args.output_dir, args.chunk_size, args.chunk_overlap, args.ocr_threshold, extract_images)
            print(f"✅ Processed {os.path.basename(pdf_path)} -> {out_path}")
        except Exception as e:
            print(f"❌ Failed to process {os.path.basename(pdf_path)}: {e}")

if __name__ == "__main__":
    main()