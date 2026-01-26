# LLM Fine-Tuning - Practical Example

## 📚 Project Overview

This repository presents a practical and educational example of fine-tuning a Large Language Model (LLM) using datasets structured in the instruction → input → output format 🤖➡️📄➡️✅.
The objective is to illustrate, in a clear and reproducible manner, the complete workflow required to adapt a pre-trained language model to a specific task or domain 🔬📊.

🎓 This project is intended for academic purposes, including learning, experimentation, and concept validation (proof of concept). It is not recommended for production use 🚫🏭.

## 🎯 Objective

- Demonstrate how to fine-tune a pre-trained LLM
- Explore the instruction tuning data format
- Apply modern techniques such as LoRA / QLoRA
- Evaluate how fine-tuning changes model behavior
- Serve as a baseline for experiments in different domains (e.g., healthcare, psychology, customer support)

## 🧠 Model Architecture and Training Setup

- Base model: google/gemma-2b-it
- Framework: Hugging Face Transformers
- Training backend: PyTorch
- Quantization: bitsandbytes (4-bit / 8-bit)

## 📊 Dataset

This work uses the [jkhedri/psychology-dataset](https://huggingface.co/datasets/jkhedri/psychology-dataset), a preference-based dataset containing paired psychological responses with contrasting interaction styles 🧠.

The dataset is shuffled at load time and supports a lightweight test mode via the DATA_SAMPLES parameter, enabling rapid validation prior to full-scale fine-tuning ⚡.

A standardized chat template is applied to all splits to match the target model’s conversational format. Only the empathetic and therapeutically appropriate responses (response_j) are selected for training, while judgmental or aggressive alternatives (response_k) are explicitly excluded 🚫.

This design ensures the model learns to produce safe, professional, and supportive psychological guidance.

## 🔧 Local Training Setup and Execution

To perform local fine-tuning, the project repository (**fine-tune-llm**) is accessed and the Python environment is initialized using [Poetry](https://python-poetry.org/) 📦.

The setup procedure consists of activating the virtual environment and installing all dependencies:

```bash
poetry shell && poetry install
```

The experiments are conducted using Jupyter notebooks, opened via VS Code 💻. The **notebooks/** directory contains inference and fine-tuning notebooks, as well as the output directory used during training:

- 01_GEMMA2B_QUICK_INFERENCE.ipynb
- 02_GEMMA2B_FINE_TUNING.ipynb
- Gemma-2b-it-Psych/ (training artifacts and logs)

This workflow enables reproducible local experimentation and model fine-tuning 🧪.

## 📊 Training Monitoring and Visualization 

**TensorBoard** is used to monitor training metrics in real time, enabling continuous inspection of the optimization process. This includes tracking the training loss, observing convergence behavior, and identifying potential instabilities during fine-tuning.

📈 Example TensorBoard Visualization

Below is an example of the training loss curve visualized using TensorBoard during the fine-tuning process:

![TensorBoard Training Example](imgs/tensorboard1.png)

Figure 1: Training loss monitored via TensorBoard during fine-tuning of the Gemma-2B model.
