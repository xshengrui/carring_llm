import os
from pdfminer.high_level import extract_text
from tqdm import tqdm
import argparse


def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容。
    """
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def process_all_pdfs(input_dir, output_dir):
    """
    处理指定目录下的所有PDF文件，并保存提取的文本。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

    # 记录已处理的文件数量，用于检查断点续传
    processed_count = 0
    if os.path.exists(os.path.join(output_dir, "processed_count.txt")):
        with open(os.path.join(output_dir, "processed_count.txt"), 'r') as f:
            processed_count = int(f.read().strip())

    # 从上一次处理停止的地方开始
    start_index = processed_count

    print(f"Starting PDF processing from index {start_index}...")

    for i, pdf_file in enumerate(tqdm(pdf_files[start_index:], desc="Processing PDFs")):
        pdf_path = os.path.join(input_dir, pdf_file)
        txt_filename = os.path.splitext(pdf_file)[0] + ".txt"
        output_txt_path = os.path.join(output_dir, txt_filename)

        if os.path.exists(output_txt_path):
            # 如果输出文件已存在，跳过处理，继续下一个
            print(f"Skipping {pdf_file}, text already extracted.")
            continue

        text = extract_text_from_pdf(pdf_path)
        if text:
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(text)

        # 更新已处理文件数量
        processed_count = start_index + i + 1
        with open(os.path.join(output_dir, "processed_count.txt"), 'w') as f:
            f.write(str(processed_count))

    print(f"Finished processing {len(pdf_files)} PDFs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pdf_dir', type=str, default="./textbooks/pdf1")
    parser.add_argument('--output_text_dir', type=str, default="./pdf/output/extracted_textbooks1")
    args = parser.parse_args()

    print(f"Input PDF directory: {args.input_pdf_dir}")
    print(f"Output text directory: {args.output_text_dir}")

    process_all_pdfs(args.input_pdf_dir, args.output_text_dir)
    print("PDF文本提取完成。")
