import json
from pathlib import Path
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

JUDGE_PROMPT = """
你是一名专业的医学护理知识问答评测员。你的任务是比较模型的回答与标准答案之间的知识要点是否一致。
请依据以下标准进行评分：

评分标准（总分0-2分）：
- 0分：回答错误
- 1分：部分回答正确
- 2分：完全回答正确

【问题】{question}
【标准答案】{answer}
【模型回答】{model_response}

评分（仅输出一个阿拉伯数字）：
"""

def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate model outputs.")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-32B")
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs to use')
    return parser

def init_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    for item in data:
        item['prompt_judge_score'] = []
    return data


def build_prompt(question, answer, model_response):
    user_prompt =  JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        model_response=model_response
    )
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text

def eval_prompt_judge(input_path, output_path):
    data = init_data(input_path)
    prompts = [build_prompt(item["question"], item["answer"], item["model_response"]) for item in data]

    for i in range(3):
        outputs = llm.generate(prompts, sampling_params)
        for idx, output in enumerate(outputs):
            text = output.outputs[0].text.strip()
            try:
                score = int(''.join(filter(str.isdigit, text)))
                if 0 <= score <= 2:
                    data[idx]['prompt_judge_score'].append(score)
            except:
                pass

    for item in data:
        if item['prompt_judge_score']:
            item['prompt_judge_score'] = sum(item['prompt_judge_score']) / len(item['prompt_judge_score'])
        else:
            item['prompt_judge_score'] = 0

    with open(output_path, 'w', encoding='utf-8') as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

def calc_and_print_avg_score(output_path: str):
    scores = []
    with open(output_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            data = json.loads(line)
            score = data.get('prompt_judge_score', None)
            if score is not None and score != -1:
                scores.append(score * 50)
    if scores:
        avg = sum(scores) / len(scores)
        print(f"平均分: {avg:.2f} (0-100分制, 样本数: {len(scores)})")
    else:
        print("未找到有效分数，无法计算平均分。")

def main():
    parser = get_parser()
    args = parser.parse_args()
    input_path = args.input_file
    output_path = args.output_file
    model_path = args.model_path
    gpus = args.gpus

    import os
    if not os.path.exists(input_path):
        print("input file 未找到")
        return 0

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    global llm
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    global sampling_params
    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, max_tokens=64
    )

    llm = LLM(
        model=model_path,
        tensor_parallel_size=gpus,
        max_num_seqs=16
    )

    eval_prompt_judge(input_path, output_path)
    calc_and_print_avg_score(output_path)

if __name__ == "__main__":
    main()
