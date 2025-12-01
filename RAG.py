#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import math

# --- LangChain RAG 组件导入 ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List

# --- RAG 数据相关配置 ---
RAG_DATA_PATH = "./pdf/output/extracted_texts"
RAG_EMBEDDING_MODEL_NAME = "./models/models--BAAI--bge-small-zh-v1.5"
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
RAG_SEARCH_K = 3

# --- RAG 增强的用户 Prompt 模板 ---
RAG_USER_PROMPT_TEMPLATE = """参考资料:
{context}

---

根据上面的参考资料，回答以下问题。如果参考资料中没有直接答案，请结合你的医学护理专业知识进行回答。请确保回答专业、准确、全面。

问题: {question}
"""

# 定义命令行参数解析函数
def parse_args():
    parser = argparse.ArgumentParser(description="批量推理脚本")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径路径")
    parser.add_argument("--gpus", type=int, default=1, help="使用的 GPU 数量")
    parser.add_argument("--batch_size", type=int, default=32, help="每次推理的批次大小")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="VLLM使用的GPU内存比例")
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192, help="VLLM批处理中的最大token数")
    return parser.parse_args()

# --- RAG Setup Function ---
def setup_rag(data_path: str, embedding_model_name: str, chunk_size: int, chunk_overlap: int):
    if not os.path.exists(data_path):
        print(f"错误：RAG数据目录 {data_path} 不存在。将仅使用模型自身知识回答。")
        return None

    print(f"开始加载和处理RAG数据来自: {data_path}")

    try:
        # 1. 加载文档
        loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader, show_progress=True)
        documents = loader.load()

        if not documents:
            print(f"警告：在 {data_path} 目录中未找到任何 .txt 文件。将仅使用模型自身知识回答。")
            return None

        print(f"加载 {len(documents)} 个文档。")

        # 2. 分割文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        print(f"分割成 {len(chunks)} 个文本块。")

        # 3. 初始化嵌入模型
        print(f"初始化嵌入模型: {embedding_model_name}")
        embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cuda'})
        print("嵌入模型初始化完成。")

        # 4. 构建向量存储
        print("开始构建向量存储 (Chroma)...")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        print("向量存储构建完成。")

        # 5. 创建检索器
        retriever = vectorstore.as_retriever(search_kwargs={"k": RAG_SEARCH_K})
        print("RAG 设置完成，检索器已就绪。")
        return retriever

    except Exception as e:
        print(f"RAG setup 过程中发生错误: {e}")
        print("将仅使用模型自身知识回答。")
        return None


def main():
    # 解析命令行参数
    args = parse_args()

    # --- RAG Setup ---
    retriever = setup_rag(RAG_DATA_PATH, RAG_EMBEDDING_MODEL_NAME, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)

    # 加载分词器
    print(f"加载tokenizer来自: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print("Tokenizer加载完成。")

    # 初始化LLM模型
    print(f"初始化LLM模型来自: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True
    )
    print("LLM模型初始化完成。")

    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        top_k=5,
        max_tokens=2048
    )

    # 确保输出文件目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 读取输入文件
    input_data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    input_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: 跳过输入文件中无效的JSON行: {line.strip()}")

    if not input_data:
        print("输入文件中没有找到任何有效的问答对。")
        return

    print(f"载入 {len(input_data)} 条问答对进行推理。")

    # 实现断点续传
    processed_ids = set()
    if os.path.exists(args.output_file):
        print(f"检测到输出文件 {args.output_file}，加载已处理的ID...")
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f_out_exist:
                for line in f_out_exist:
                    try:
                        item = json.loads(line)
                        if "id" in item:
                            processed_ids.add(item["id"])
                    except json.JSONDecodeError:
                        pass
            print(f"已加载 {len(processed_ids)} 条已处理的问答对ID。")
        except Exception as e:
            print(f"加载现有输出文件出错: {e}. 将忽略现有内容并从头开始处理未完成的样本。")
            processed_ids = set()

    # 过滤掉已处理的样本
    items_to_process = [item for item in input_data if item["id"] not in processed_ids]

    if not items_to_process:
        print("所有问答对都已处理。")
        return

    print(f"剩余 {len(items_to_process)} 条问答对待处理。")

    # 存储所有Prompt
    all_prompts = []
    # 存储对应的原始item，用于结果输出
    original_items_for_batch = []

    # --- 准备带RAG上下文的Prompt ---
    print("准备带RAG上下文的Prompt...")
    for item in tqdm(items_to_process, desc="准备Prompt"):
        question = item.get("question", "")

        # 1. RAG 检索
        context = ""
        if retriever:
            try:
                retrieved_docs: List[Document] = retriever.get_relevant_documents(question)
                # 将检索到的文档内容拼接起来作为上下文
                context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
            except Exception as e:
                 print(f"Warning: 对问题 ID={item.get('id')} 进行RAG检索时发生错误: {e}. 跳过RAG，仅使用模型知识。")
                 context = "" # 清空上下文，退化为非RAG模式

        # 2. 构建 RAG 增强的用户 Prompt
        user_prompt_content = RAG_USER_PROMPT_TEMPLATE.format(context=context, question=question)

        # 3. 构建消息格式
        messages = [
            {"role": "user", "content": user_prompt_content}
        ]

        # 使用Qwen的chat template生成最终Prompt
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except Exception as e:
             print(f"Error applying chat template for item ID={item.get('id')}: {e}")
             prompt = f"{RAG_USER_PROMPT_TEMPLATE}\n{user_prompt_content}"

        all_prompts.append(prompt)
        original_items_for_batch.append(item)

    # 以追加模式写入结果，确保断点续传时不会覆盖
    with open(args.output_file, 'a', encoding='utf-8') as f_out:
        # 批量处理所有Prompt
        num_batches = math.ceil(len(all_prompts) / args.batch_size)

        # tqdm 用于显示批次处理进度
        for i in tqdm(range(num_batches), desc="批量推理进度"):
            batch_start = i * args.batch_size
            batch_end = min((i + 1) * args.batch_size, len(all_prompts))

            current_batch_prompts = all_prompts[batch_start:batch_end]
            # 获取当前批次对应的原始 item 对象
            current_batch_original_items = original_items_for_batch[batch_start:batch_end]

            if not current_batch_prompts:
                continue

            try:
                # 批量生成回答
                results = llm.generate(current_batch_prompts, sampling_params)

                # 处理结果
                for j, result in enumerate(results):
                    original_item = current_batch_original_items[j]

                    # VLLM 生成结果可能有多项，这里取第一个
                    if result.outputs:
                        generated_text = result.outputs[0].text.strip()
                    else:
                         generated_text = ""

                    output_item = {
                        "id": original_item.get("id"),
                        "question": original_item.get("question", ""),
                        "answer": original_item.get("answer", ""),
                        "model_response": generated_text
                    }
                    f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                    f_out.flush()

            except Exception as e:
                print(f"\nError during VLLM batch generation (batch {i+1}/{num_batches}): {e}")

    print(f"推理完成。所有结果已保存到 {args.output_file}。")

if __name__ == "__main__":
    main()
