from trl import GRPOConfig, GRPOTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from bert_score import BERTScorer
import re
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["BERTSCORE_RESOURCE_DIR"] = "./bert-zh-model"

# 系统提示调整建议（符合临床步骤）
cot_system_prompt = """你是一个护理学专家，请按以下步骤回答问题：
1. 临床场景分析：识别患者的关键症状和护理需求
2. 理论依据：关联相关护理学理论和操作规范
3. 推理过程：用<reasoning>...</reasoning>包裹逻辑推理
4. 最终答案：用<answer>...</answer>包裹明确结论"""

# 强化学习奖励计算器（适配GRPOTrainer接口）
class CotRewardFunction:
    __name__ = "CotRewardFunction"  # 添加类属性
    def __init__(self, references):
        self.scorer = BERTScorer(
            model_type="/inspire/hdd/project/socialscience/xialingying041-summer-041/project/bert-zh-model",  # 直接使用本地路径作为 model_type
            num_layers=12,                  # bert-base-chinese 的层数
        )
        self.reference_dict = {ref["question"]: ref["answer"] for ref in references}

    def __call__(self, prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # 格式奖励（30%）
            format_score = self._calc_format_score(completion)

            # 内容奖励（70%）
            content_score = self._calc_content_score(prompt, completion)

            rewards.append(0.3*format_score + 0.7*content_score)
        return rewards

    def _calc_format_score(self, text):
        """严格格式验证"""
        reasoning_blocks = re.findall(r'<reasoning>\s*(.*?)\s*</reasoning>', text, re.DOTALL)
        answer_blocks = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)

        if len(reasoning_blocks) !=1 or len(answer_blocks)!=1:
            return 0.0  # 格式错误惩罚

        answer_len = len(answer_blocks[0].strip())
        reasoning_len = len(reasoning_blocks[0].strip())
        if reasoning_len >= 30 and answer_len >= 50:
            return 1.0  # 完全符合要求
        elif reasoning_len >= 15 or answer_len >= 25:  # 部分符合
            return 0.5
        else:
            return 0.0  # 即使格式正确但内容过短

    def _find_reference(self, prompt):
        """增强型答案匹配"""
        # 双重匹配策略
        question_from_prompt = re.findall(
            r"(问题|Question)[：:]?\s*(.*?)(?=\n|$)",
            prompt,
            re.DOTALL
        )
        # 获取最后一个匹配的问题（假设最新问题在最后）
        current_question = question_from_prompt[-1][1].strip() if question_from_prompt else ""

        # 分层查找策略
        reference_answer = self.reference_dict.get(current_question, None)
        if not reference_answer:
            # 模糊匹配：去除标点后的部分匹配
            clean_question = re.sub(r"[^\w\u4e00-\u9fff]", "", current_question)
            for q, a in self.reference_dict.items():
                if clean_question in re.sub(r"[^\w\u4e00-\u9fff]", "", q):
                    return a
            # 最终保底返回
            return "[未找到匹配答案]"
        return reference_answer

    def _calc_content_score(self, prompt, completion):
        """增强的内容评分"""
        gen_answer = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if not gen_answer:
            return 0.0

        ref_answer = self._find_reference(prompt)
        if not ref_answer.strip():
            print(f"警告：未找到问题对应的参考答案 | Prompt: {prompt[:100]}...")
            return 0.0

        # 过滤空内容
        gen_text = gen_answer[0].strip()
        if not gen_text:
            return 0.0

        # 执行评分
        try:
            _, _, F1 = self.scorer.score([gen_text], [ref_answer])
            return F1.mean().item()
        except Exception as e:
            print(f"评分错误：{str(e)}")
            return 0.0


tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/project/socialscience/xialingying041-summer-041/textbooks/model_results/1k/weights/checkpoint-285")

# 数据预处理
def format_prompt(example):
    messages = [
        {"role": "system", "content": cot_system_prompt},
        {"role": "user", "content": example["question"]}
    ]
    return {"prompt": tokenizer.apply_chat_template(messages, tokenize=False)}

# 主训练流程
def train_rl_cot():
    # 加载微调后的模型
    model = AutoModelForCausalLM.from_pretrained(
        "/inspire/hdd/project/socialscience/xialingying041-summer-041/textbooks/model_results/1k/weights/checkpoint-285",
        # torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/project/socialscience/xialingying041-summer-041/textbooks/model_results/1k/weights/checkpoint-285")

    # 加载数据集
    dataset = load_dataset("json", data_files={"train": "/inspire/hdd/project/socialscience/xialingying041-summer-041/strategyE+SFT+GRPO+pdf+textbook/data.jsonl"})["train"]
    dataset = dataset.map(format_prompt)

    # 初始化奖励函数
    reward_func = CotRewardFunction(dataset)


    grpo_config = GRPOConfig(
        output_dir="./rl_results",
        logging_dir="logs",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        optim="adafactor",
        learning_rate=1e-7,
        beta=0.2,
        lr_scheduler_type="cosine",
        num_generations=4,
        max_completion_length=256,
        temperature=0.4,
        top_p=0.9,
        loss_type="bnpo",
        num_iterations=2,
        epsilon=0.2,
        epsilon_high=0.28,
        scale_rewards=False,
        logging_steps=10,
        save_steps=500
    )

    torch.cuda.empty_cache()
    torch.backends.cuda.max_split_size_mb = 128


    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_func,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 开始训练
    trainer.train()
    trainer.save_model("./rl_cot_model")

if __name__ == "__main__":
    train_rl_cot()
