#!/usr/bin/env python3
"""
PDF Reader for NanoLM Research Papers
"""
import pdfplumber
import os
from pathlib import Path

def read_pdf_content(pdf_path):
    """Extract text content from PDF"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def main():
    pdf_dir = Path("pdf")
    
    if not pdf_dir.exists():
        print("PDF directory not found")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        print(f"\n{'='*80}")
        print(f"Reading: {pdf_file.name}")
        print(f"{'='*80}")
        
        content = read_pdf_content(pdf_file)
        if content:
            # Save extracted text
            txt_file = pdf_file.with_suffix('.txt')
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Content saved to: {txt_file}")
            print("\nFirst 2000 characters:")
            print("-" * 50)
            print(content[:2000])
            print("-" * 50)
        else:
            print(f"Failed to extract content from {pdf_file}")

if __name__ == "__main__":
    main()
