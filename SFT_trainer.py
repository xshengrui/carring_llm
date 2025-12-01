from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch
from datasets import load_dataset
import os
from trl import SFTTrainer, SFTConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(
    '/inspire/hdd/project/socialscience/public/models/Qwen3-8B',
    trust_remote_code=True,
    padding_side="right"
)
model = AutoModelForCausalLM.from_pretrained(
    '/inspire/hdd/project/socialscience/public/models/Qwen3-8B',
    device_map='auto',
    torch_dtype=torch.bfloat16,
    use_cache=False  # 模型加载时禁用缓存
)

# 确保pad_token配置正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id


def build_prompt_strategy_e(question, tokenizer):
    """策略 E：结合 CoT 与专家角色设定 (强化版 CoT)"""
    user_prompt = (
        "你是一名资深的医疗护理专家。请你针对以下护理问题，首先进行详细的专业思考，包括但不限于问题的关键点、可能涉及的护理原则、相关医学知识等。然后，基于你的思考，给出全面、准确、专业的最终回答。\n"
        f"【问题】{question}\n"
        "【专业思考过程】"
    )
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True, # 如果分词器支持该参数
    )
    return text

def create_prompt(example):
    text = build_prompt_strategy_e(example["question"], tokenizer)
    example["prompt"] = text
    return example

def process_func(example):
    MAX_LENGTH = 1024
    instruction = tokenizer(example["prompt"],
                          add_special_tokens=False,
                          truncation=True,
                          max_length=MAX_LENGTH)

    response = tokenizer(f"{example['answer']}<|eot_id|>",
                       add_special_tokens=False,
                       truncation=True,
                       max_length=MAX_LENGTH)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100]*len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

print("-----Load SFT dataset-----")
dataset = load_dataset(
    "json",
    data_files={
        "train": "/inspire/hdd/project/socialscience/xialingying041-summer-041/project/data/new_train_data/combined_data.jsonl"
    }
)["train"]

print("-----train_test_split-----")
actual_sample_size = len(dataset)
sampled_dataset = dataset.shuffle(seed=42).select(range(actual_sample_size))
train_test_split = sampled_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print("-----create prompt-----")
train_dataset = train_dataset.map(create_prompt)
eval_dataset = eval_dataset.map(create_prompt)

print("-----transfer to tokenized id-----")
tokenized_train = train_dataset.map(
    process_func,
    remove_columns=train_dataset.column_names,
    batched=False
)
tokenized_eval = eval_dataset.map(
    process_func,
    remove_columns=eval_dataset.column_names,
    batched=False
)

# 修正后的训练参数
args = SFTConfig(
    output_dir="./weights",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_dir="./logs",
    logging_steps=20,
    eval_steps=30,
    num_train_epochs=10,
    save_steps=300,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    gradient_checkpointing=False,
    seed=42,
    # bf16=True,
    max_length=1024,
    dataset_text_field="text",
    padding_free=False
)

# 关键修改点：使用 processing_class 传递分词器
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    processing_class=tokenizer
)

print("-----Starting Training-----")
trainer.train()
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
print("-----SFT succeeded!-----")
