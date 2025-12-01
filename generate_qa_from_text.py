import argparse
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time

# 定义数据构造的Prompt模板
# 优化后的Prompt，明确要求生成10个问题
QA_GENERATION_PROMPT = """
你是一名专业的护理知识问答出题人。你的任务是根据给定的文章内容，生成 **10个** 高难度、专业性强、贴近临床实操或考试场景的护理问题，以及每个问题对应的直接、准确、全面的答案。

问题应关注：
1. 护理操作的具体步骤或注意事项。
2. 疾病的评估、诊断、治疗或护理干预措施。
3. 药物的使用、副作用或管理。
4. 特定人群（如孕妇、老年人、儿童）的护理特点。
5. 紧急情况下的应对策略。
6. 考试中可能出现的知识点，例如定义、分类、原理或鉴别。

答案必须：
1. 直接从文章中提取或高度概括，确保真实性和权威性。
2. 详细、准确，足以解决问题。
3. 如果文章内容不足以完全回答问题，请在答案中明确指出，但仍尝试给出文章中能提供的信息。

请确保所有问题和答案都使用清晰、专业的中文。

文章内容：
{text_content}

请严格遵循以下JSON数组格式输出结果，不要包含任何额外信息或解释：
[
  {{
    "question": "在这里填写生成的护理问题1",
    "answer": "在这里填写生成的准确答案1"
  }},
  {{
    "question": "在这里填写生成的护理问题2",
    "answer": "在这里填写生成的准确答案2"
  }},
  // ... 以此类推，直到第10个问答对
  {{
    "question": "在这里填写生成的护理问题10",
    "answer": "在这里填写生成的准确答案10"
  }}
]
"""

def parse_args():
    parser = argparse.ArgumentParser(description="根据文本生成问答对脚本")
    parser.add_argument("--input_text_dir", type=str, required=True, help="包含提取文本的目录")
    parser.add_argument("--output_jsonl_file", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="用于数据构造的模型路径")
    parser.add_argument("--gpus", type=int, default=1, help="使用的 GPU 数量")
    parser.add_argument("--batch_size", type=int, default=1, help="批量推理的批次大小，每个PDF生成10个问题，建议单次处理一个PDF")
    parser.add_argument("--num_qa_per_doc", type=int, default=10, help="每个文档生成问答对的数量")
    parser.add_argument("--max_retries", type=int, default=3, help="生成失败时的最大重试次数")
    parser.add_argument("--start_idx", type=int, default=0, help="从排序后的PDF文件列表的第几（0-indexed）个文件开始处理")
    parser.add_argument("--end_idx", type=int, default=-1, help="处理到排序后的PDF文件列表的第几（0-indexed）个文件（包含此文件）。-1表示处理到最后一个文件")
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.gpus)

    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        max_tokens=2048,      # 增加 max_tokens 以适应生成多个QA对的需求
        stop=["\n]"] # 尝试更精准的停止条件，例如在JSON数组闭合的换行符和右括号处停止
    )

    extracted_text_files_list = []
    # 按照文件名排序，确保每次运行的顺序一致
    text_filenames = sorted([f for f in os.listdir(args.input_text_dir) if f.endswith(".txt")])

    # 根据start_idx和end_idx筛选文件
    start_index = args.start_idx
    end_index = args.end_idx if args.end_idx != -1 else len(text_filenames) - 1 # -1表示到最后一个文件

    if start_index < 0 or start_index >= len(text_filenames):
        print(f"错误：start_idx ({start_index}) 超出文件列表范围 (0 - {len(text_filenames) - 1})。")
        return
    if end_index < start_index or end_index >= len(text_filenames):
        print(f"警告：end_idx ({end_index}) 超出文件列表范围或小于start_idx。将处理到最后一个文件 ({len(text_filenames) - 1})。")
        end_index = len(text_filenames) - 1

    files_to_process = text_filenames[start_index : end_index + 1]

    for filename in files_to_process:
        file_path = os.path.join(args.input_text_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.strip(): # 避免空文件
                extracted_text_files_list.append((filename, content))

    if not extracted_text_files_list:
        print(f"在 {args.input_text_dir} 中指定范围内没有找到任何文本文件，请确保PDF已成功提取或索引范围正确。")
        return

    # 检查输出文件是否存在，如果存在则读取已有的ID，实现断点续传的ID管理
    output_dir = os.path.dirname(args.output_jsonl_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_id = 0
    if os.path.exists(args.output_jsonl_file):
        print(f"检测到现有输出文件: {args.output_jsonl_file}。尝试加载已有的问答对以确定起始ID...")
        try:
            with open(args.output_jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if "id" in item:
                            current_id = max(current_id, item["id"] + 1)
                    except json.JSONDecodeError:
                        print(f"Warning: 现有文件中存在无效JSON行: {line.strip()}")
        except Exception as e:
            print(f"读取现有文件时发生错误: {e}")
        print(f"已加载 {current_id} 作为新的问答对起始ID。")

    print(f"将处理从索引 {start_index} 到 {end_index} 的 {len(extracted_text_files_list)} 篇文本文章。正在生成问答对...")

    # 以追加模式打开输出文件
    with open(args.output_jsonl_file, 'a', encoding='utf-8') as f_out:
        for doc_idx, (filename, text_content) in enumerate(tqdm(extracted_text_files_list, desc="生成问答对")):
            # 限制每个文本块的大小
            if len(text_content) > 6000: # 根据模型最大上下文窗口调整
                text_content = text_content[:6000] # 简单截断

            generated_count_for_doc = 0
            retries = 0

            while generated_count_for_doc < args.num_qa_per_doc and retries < args.max_retries:
                prompt = QA_GENERATION_PROMPT.format(text_content=text_content)
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )

                try:
                    results = llm.generate([formatted_prompt], sampling_params)
                    generated_text = results[0].outputs[0].text.strip()

                    # 尝试修复JSON格式
                    # 如果生成内容不是以[开头，则尝试添加
                    if not generated_text.startswith("["):
                        generated_text = "[" + generated_text
                    # 如果生成内容不是以]结尾，则尝试添加
                    if not generated_text.endswith("]"):
                        generated_text += "]"

                    qa_list = json.loads(generated_text)

                    if not isinstance(qa_list, list):
                        print(f"Warning: 模型生成的不是JSON数组，重试。文件: {filename}, 片段: {text_content[:100]}...")
                        retries += 1
                        continue

                    valid_qa_in_batch = 0
                    for qa_pair in qa_list:
                        if "question" in qa_pair and "answer" in qa_pair:
                            output_item = {
                                "id": current_id,
                                "question": qa_pair["question"].strip(),
                                "answer": qa_pair["answer"].strip()
                            }
                            f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                            current_id += 1
                            valid_qa_in_batch += 1
                        else:
                            print(f"Warning: 模型生成的JSON对象缺少'question'或'answer'字段。文件: {filename}, JSON: {qa_pair}")

                    generated_count_for_doc += valid_qa_in_batch

                    if generated_count_for_doc < args.num_qa_per_doc:
                        print(f"文档 '{filename}' 生成了 {valid_qa_in_batch} 个QA，目标 {args.num_qa_per_doc}。正在重试... (已重试 {retries+1} 次)")
                        retries += 1

                except json.JSONDecodeError:
                    print(f"Warning: 模型生成了无效的JSON输出，重试。文件: {filename}, 片段: {text_content[:100]}...\n{generated_text}")
                    retries += 1
                except Exception as e:
                    print(f"Error generating QA for '{filename}': {e}. 重试... (已重试 {retries+1} 次)")
                    retries += 1

            if generated_count_for_doc < args.num_qa_per_doc:
                print(f"警告：文档 '{filename}' 未能生成目标数量 {args.num_qa_per_doc} 个问答对，仅生成了 {generated_count_for_doc} 个。")
            else:
                print(f"文档 '{filename}' 成功生成了 {generated_count_for_doc} 个问答对。")

    print(f"数据构造完成。总共生成了 {current_id} 条问答对。")

if __name__ == "__main__":
    main()
