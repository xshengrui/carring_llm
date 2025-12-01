import os
import re
from docx import Document

def extract_text_from_docx(docx_path, txt_path):
    """从单个 .docx 文件提取文本并保存到 .txt 文件，同时删除文字之间的空格，保留换行符"""
    doc = Document(docx_path)

    text = []
    for para in doc.paragraphs:
        para_text = para.text.strip()  # 去掉首尾空格
        # 删除文字之间的所有空格
        para_text = re.sub(r'\s+', '', para_text)

        if para_text:
            text.append(para_text)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(text))

    print(f"文本已成功提取到 {txt_path}")

def process_all_docx_in_folder(folder_path, output_folder, record_file):
    """处理文件夹中的所有 .docx 文件并支持断点续传"""
    # 获取已处理文件的记录
    processed_files = set()
    if os.path.exists(record_file):
        with open(record_file, 'r', encoding='utf-8') as record_f:
            processed_files = set(record_f.read().splitlines())

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            docx_path = os.path.join(folder_path, filename)
            txt_filename = filename.replace('.docx', '.txt')
            txt_path = os.path.join(output_folder, txt_filename)

            # 跳过已处理的文件
            if filename in processed_files:
                print(f"跳过已处理的文件: {filename}")
                continue

            # 提取文本并保存到 .txt 文件
            extract_text_from_docx(docx_path, txt_path)

            # 记录已处理的文件
            with open(record_file, 'a', encoding='utf-8') as record_f:
                record_f.write(f"{filename}\n")

# 设置文件夹路径和记录文件
folder_path = './word'  # .docx 文件所在的文件夹
output_folder = './txt'  # 输出的 .txt 文件所在的文件夹
record_file = './word/processed_files.txt'  # 记录已处理文件的文件

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 处理所有 .docx 文件
process_all_docx_in_folder(folder_path, output_folder, record_file)
