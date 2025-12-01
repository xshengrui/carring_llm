# 大模型护理问题应答能力构建项目

本项目是一个专注于护理领域的大语言模型应用，旨在通过监督微调(SFT)和强化学习(RL)技术提升模型在护理问答场景下的表现。

## 项目结构

本项目包含以下核心模块：

1. **eval.py** - 评估模型输出，对模型回答进行评分
2. **inference_x.py** - 推理脚本，支持多种提示策略
3. **RAG.py** - 基于检索增强生成(RAG)的推理实现
4. **rl_trainer.py** - 强化学习训练脚本
5. **SFT_trainer.py** - 监督微调训练脚本
6. **extract_pdf_text.py** - PDF文本提取工具
7. **generate_qa_from_text.py** - 从文本生成问答对
8. **word2txt.py** - Word文档转文本工具
9. **combine.py** - 合并数据工具
10. **rejection_sampling.py** - 拒绝采样工具

## 使用方法

### 1. 数据准备

首先，使用以下工具准备训练数据：

- 使用 `extract_pdf_text.py` 提取PDF文档中的文本内容
- 使用 `word2txt.py` 将Word文档转换为纯文本
- 使用 `generate_qa_from_text.py` 从文本生成护理问答对
- 使用 `combine.py` 合并多个数据源
- 使用 `rejection_sampling.py` 对生成的问答对进行质量过滤

### 2. 模型训练

#### 监督微调(SFT)

```bash
python SFT_trainer.py
```

#### 强化学习训练(RL)

```bash
python rl_trainer.py
```

### 3. 模型推理

#### 基本推理

```bash
python inference_x.py \
    --input_file 输入文件路径 \
    --output_file 输出文件路径 \
    --model_path 模型路径 \
    --gpus GPU数量
```

#### RAG推理

```bash
python RAG.py \
    --input_file 输入文件路径 \
    --output_file 输出文件路径 \
    --model_path 模型路径 \
    --gpus GPU数量 \
    --batch_size 批量大小 \
    --gpu_memory_utilization GPU内存利用率 \
    --max_num_batched_tokens 最大批处理token数
```

### 4. 模型评估

```bash
python eval.py \
    --input_file 待评估文件路径 \
    --output_file 评估结果路径 \
    --model_path 评估模型路径 \
    --gpus GPU数量
```

## 技术特点

1. **多策略提示**：支持多种提示策略，包括CoT、Few-shot等
2. **RAG增强**：集成检索增强生成技术，提高回答准确性
3. **强化学习**：使用GRPO算法进行强化学习训练
4. **断点续传**：支持处理过程中的断点续传功能
5. **批量处理**：支持批量推理和评估，提高处理效率

## 依赖环境

- Python 3.8+
- PyTorch
- Transformers
- vLLM
- LangChain
- TRL
- BERTScore

## 注意事项

1. 请根据实际环境修改脚本中的路径配置
2. GPU资源使用量较大，请确保有足够的显存

## 贡献者

毛宣杰、许圣瑞、潘亚宁-SII009-2025.05
