import argparse
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 评分Prompt模板
SCORE_PROMPT = """
以下是一个护理问题及其回答。请根据以下六个标准对该问答进行评分，评分区间为1-10，其中1表示极差，10表示非常优秀。

1. **准确性**（Accuracy）：答案是否准确，是否符合护理实践中的标准和知识？
2. **相关性**（Relevance）：答案是否直接回答了问题，是否提供了解决方案或实用的护理建议？
3. **清晰度**（Clarity）：答案是否表达清晰，易于理解？是否适合患者或护理人员阅读？
4. **自然性**（Naturalness）：答案是否流畅自然，是否显得不机械或过于公式化？
5. **完整性**（Completeness）：答案是否涵盖了问题的所有方面，是否遗漏重要信息？
6. **多样性**（Diversity）：答案是否能够涵盖护理领域的多样性，避免单一或过于狭窄的视角？

请按格式返回评分，格式如下：
{
  "accuracy": 8,
  "relevance": 9,
  "clarity": 7,
  "naturalness": 9,
  "completeness": 8,
  "diversity": 7
}
"""

def parse_args():
    parser = argparse.ArgumentParser(description="评估已生成的护理问答对并打分")
    parser.add_argument("--input_jsonl", type=str, required=True, help="输入的包含问答对的jsonl文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出评分结果的目录")
    parser.add_argument("--model_path", type=str, required=True, help="用于评分的模型路径")
    parser.add_argument("--gpus", type=int, default=1, help="使用的 GPU 数量")
    return parser.parse_args()

def score_qa_pairs(qa_pairs, llm, tokenizer, sampling_params):
    scored_qa_pairs = []
    for qa_pair in qa_pairs:
        prompt = SCORE_PROMPT + "\n问题：\n" + qa_pair["question"] + "\n答案：\n" + qa_pair["answer"]
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # 进行模型推理
        result = llm.generate([formatted_prompt], sampling_params)
        scored_qa_pair = result[0].outputs[0].text.strip()

        try:
            # 解析返回的评分
            score = json.loads(scored_qa_pair)
            scored_qa_pairs.append({
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                "score": score
            })
        except json.JSONDecodeError:
            print(f"Warning: 模型评分生成失败：\n{scored_qa_pair}")

    return scored_qa_pairs

def main():
    args = parse_args()

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.gpus)

    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=0.8,    # 提高温度，鼓励生成更复杂的问答
        top_p=0.9,
        top_k=50,
        max_tokens=1024,      # 增加max_tokens以适应更长、更详细的答案
        stop=["}}"] # 严格控制输出格式，在JSON闭合括号处停止
    )

    # 读取输入的jsonl文件
    with open(args.input_jsonl, 'r', encoding='utf-8') as f_in:
        qa_pairs = [json.loads(line.strip()) for line in f_in.readlines()]

    if not qa_pairs:
        print(f"在 {args.input_jsonl} 中没有找到任何问答对。")
        return

    # 检查输出目录是否存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"读取了 {len(qa_pairs)} 个问答对。正在进行评分...")

    # 评分每个生成的问答对
    scored_qa_pairs = score_qa_pairs(qa_pairs, llm, tokenizer, sampling_params)

    # 保存评分结果
    output_jsonl_file = os.path.join(args.output_dir, "scored_qa_pairs.jsonl")
    with open(output_jsonl_file, 'w', encoding='utf-8') as f_out:
        for scored_qa in scored_qa_pairs:
            f_out.write(json.dumps(scored_qa, ensure_ascii=False) + '\n')

    print(f"评分完成，结果已保存到 {output_jsonl_file}")

if __name__ == "__main__":
    main()
