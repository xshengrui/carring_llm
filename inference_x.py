import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="推理脚本")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--gpus", type=int, default=1, help="使用的 GPU 数量")
    return parser.parse_args()

####
# 这是一个示例infrence.py脚本，用于输入指定格式的 JSONL 文件，调用 Qwen3-8B 模型进行推理，并将结果保存到输出文件中。
# 大家需要根据自己的技术方案，如检索增强、多智能体系统等，自行实现推理逻辑。该脚本仅用于给出遵循输入输出格式的简单示例。
# 建议修改成batch inference的形式
####

# --- 提示词构建函数 ---
def build_prompt_default(question, tokenizer):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt

def build_prompt_strategy_a(question, tokenizer):
    """策略 A：明确角色与指令"""
    user_prompt = (
        "你是一名专业的医疗护理专家。请根据你的专业知识，准确、详细地回答以下护理问题。\n"
        f"【问题】{question}"
    )
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text

def build_prompt_strategy_b(question, tokenizer):
    """策略 B：引入 CoT (Chain-of-Thought) 思维链"""
    user_prompt = (
        "你是一名专业的医疗护理专家。请思考以下问题，并逐步推理，最终给出准确、详细的护理回答。\n"
        f"【问题】{question}\n"
        "【思考过程】"
    )
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True, # Qwen模型可能支持enable_thinking来激活CoT
    )
    return text

def build_prompt_strategy_c(question, tokenizer):
    """策略 C：强调简洁与关键信息"""
    user_prompt = (
        "请你作为一名经验丰富的护理人员，针对以下问题，提供简洁但包含所有关键信息的回答。不要包含无关的细节。\n"
        f"【问题】{question}"
    )
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text

# 假设有以下示例数据（通常需要从实际数据集中选取）
# 注意：在实际使用时，这个示例数据应该来自你的验证集或训练集，并且是高质量的。
example_qa_pair = {
    "question": "Ⅱ型呼衰病人应用高流量湿化氧疗时, 氧流量应设置为多少?",
    "answer": "氧流量在30L/min时,病人PaCO2有较大改善,建议最低流量应设置为 30L/min。"
}

def build_prompt_strategy_d(question, tokenizer, example_qa=example_qa_pair):
    """策略 D：结合示例 (Few-shot prompting)"""
    user_prompt = (
        "以下是一个护理问题及其专业回答示例：\n"
        f"【示例问题】{example_qa['question']}\n"
        f"【示例回答】{example_qa['answer']}\n\n"
        "请参考上述示例的风格和专业程度，准确回答以下新的护理问题。\n"
        f"【问题】{question}"
    )
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text

def build_prompt_strategy_e(question, tokenizer):
    """策略 E：结合 CoT 与专家角色设定 (强化版 CoT)"""
    user_prompt = (
        "你是一名资深的医疗护理专家。请你针对以下护理问题，首先进行详细的专业思考，包括但不限于问题的关键点、可能涉及的护理原则、相关医学知识等。然后，基于你的思考，给出全面、准确、专业的最终回答。\n"
        f"【问题】{question}\n"
        "【专业思考过程】" # 更明确地引导专业思考
    )
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True, # 启用CoT
    )
    return text

def build_prompt_strategy_f(question, tokenizer):
    """策略 F：多步骤思考与结构化输出 (ToT 简化版)"""
    user_prompt = (
        "你是一名严谨的医疗护理顾问。请严格按照以下步骤思考并回答护理问题：\n"
        "1. **识别核心问题**：明确用户问题的核心诉求和关键信息。\n"
        "2. **关联知识点**：思考与核心问题相关的护理学、医学理论、指南或临床实践经验。\n"
        "3. **制定初步回答**：根据关联知识点，构思一个初步的、全面的回答草稿。\n"
        "4. **审查与优化**：检查初步回答的准确性、完整性和专业性，并进行必要的修正和完善。\n"
        "最后，以清晰、条理分明的格式给出最终回答。\n"
        f"【问题】{question}\n"
        "【思考与回答】" # 引导模型按照步骤输出
    )
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True, # 启用CoT
    )
    return text

def build_prompt_strategy_g(question, tokenizer):
    """策略 G：强调准确性与安全性 (医疗领域特化)"""
    user_prompt = (
        "你是一名专业的医疗护理人工智能助理，你的首要任务是提供准确、安全、负责任的护理知识。请在回答以下问题时，务必做到信息正确无误，避免任何可能误导或造成风险的表述。\n"
        "请先详细思考，确保理解问题的所有方面，并调动所有相关专业知识。如果信息存在不确定性或需要更具体的临床判断，请在回答中明确指出。\n"
        f"【问题】{question}\n"
        "【思考与回答】"
    )
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True, # 启用CoT
    )
    return text
# --- 提示词构建函数结束 ---


def main():
    args = parse_args()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # 初始化LLM模型，设置GPU并行数量
    llm = LLM(model=args.model_path, tensor_parallel_size=args.gpus)

    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=0.6,    # 控制生成文本的随机性
        top_p=0.95,        # 控制累积概率阈值
        top_k=20,          # 控制每次选择的候选词数量
        max_tokens=32768   # 生成文本的最大长度
    )

    # 读取输入文件
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f]  # 将每行JSONL数据解析为Python对象

    prompts = []
    # 遍历处理每个输入样本，构建所有prompt
    for item in input_data:
        question = item.get("question", "")

        # 选择你想要测试的提示词策略
        # 默认策略
        # prompts.append(build_prompt_default(question, tokenizer))

        # 策略 A：明确角色与指令
        # prompts.append(build_prompt_strategy_a(question, tokenizer))

        # 策略 B：引入 CoT 思维链
        # prompts.append(build_prompt_strategy_b(question, tokenizer))

        # 策略 C：强调简洁与关键信息
        # prompts.append(build_prompt_strategy_c(question, tokenizer))

        # 策略 D：结合示例 (Few-shot prompting) - 需要 example_qa_pair
        # prompts.append(build_prompt_strategy_d(question, tokenizer))

        # 策略 E：结合 CoT 与专家角色设定 (强化版 CoT)
        prompts.append(build_prompt_strategy_e(question, tokenizer))

        # 策略 F：多步骤思考与结构化输出 (ToT 简化版)
        # prompts.append(build_prompt_strategy_f(question, tokenizer))

    print(f"Total prompts to generate: {len(prompts)}")

    # 批量生成
    outputs = llm.generate(prompts, sampling_params)

    # 将vLLM的输出映射回原始的input_data
    # 由于prompt可能重复，这里需要更稳健的映射方式
    # outputs是按照输入prompts列表的顺序返回的

    # 存储输出结果的列表
    processed_outputs = []

    for i, item in enumerate(input_data):
        qid = item.get("id")
        question = item.get("question", "")
        answer = item.get("answer", "")

        # 确保输出结果与原始输入对应
        # 注意：这里假设outputs的顺序与prompts的顺序一致
        # vLLM默认会保持请求的顺序，所以可以通过索引直接对应
        generated_text = outputs[i].outputs[0].text.strip()

        output_item = {
            "id": qid,
            "question": question,
            "answer": answer,
            "model_response": generated_text
        }
        processed_outputs.append(output_item)

    # 将结果写入输出文件
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in processed_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 程序入口点
if __name__ == "__main__":
    main()
