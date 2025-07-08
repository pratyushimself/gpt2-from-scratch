# 🧠 Building a Transformer Model from Scratch (GPT-style)

This project walks through the full pipeline of building and training a GPT-like transformer model from scratch — starting with a custom tokenizer, followed by a small transformer architecture, and finally training it on a subset of OpenWebText data.

---

## 🎯 Purpose

The goal of this project is **not just to fine-tune an existing GPT model**, but to **understand and implement the entire process** behind how transformer models like GPT work — from tokenization to training to text generation.

This was done using a simplified architecture and smaller dataset, so output quality isn’t the main focus. Instead, the project is meant to help you:

- Learn how GPT-style models are trained from raw data.
- Build and customize your own tokenizer.
- Structure a transformer training workflow using PyTorch and HuggingFace.

---

## 🔍 Key Concepts

- **Custom Tokenizer**: Built using Byte-Pair Encoding (BPE) from raw text data.
- **Model Architecture**: A compact GPT-2-style model with 4 layers and 256-dimensional embeddings.
- **Training**: Performed chunk-by-chunk on tokenized data (~20 `.pt` chunks for demo).
- **Resume Support**: Training can resume from checkpoints (e.g., `checkpoint-56000`).
- **Generation**: Output is generated using the trained model and tokenizer.

---
FLOW CHART

![FLOW chart](https://github.com/user-attachments/assets/af081798-b8a1-4705-9b1c-2c511898dfde)

---

🔁 Training in Progress

Training Screenshot 1(![Screenshot 2025-07-09 000534](https://github.com/user-attachments/assets/ac8ac04c-9a35-4f19-a9de-f9c40d15514d)
)

Training Screenshot 2(![Screenshot 2025-07-09 000828](https://github.com/user-attachments/assets/48009cba-5a93-479f-8c8e-2f03d4a54683)
)
*Training completed and model saved locally.*

---

## 🧪 Data & Personalization

> ⚠️ _If you're wondering why the generated text might feel generic... it's all in the data._

To make the model generate meaningful, context-aware outputs, **the dataset should reflect the kind of content you want the model to generate**. That could be:

- Your own writing (e.g., journals, text messages, blogs).
- Conversational data in your voice.
- Structured knowledge or domain-specific content.

This version was trained on a small chunked subset of OpenWebText — **just 20 training files out of a larger dataset**, for demonstration purposes.

**Bigger and more authentic data leads to better results.**

---
## 🛠️ Project Structure

- `gpt2_custom_model/` — Trained model & checkpoints  
- `tokenizer_gpt2_custom/` — Custom tokenizer files  
- `token_chunks/` — Training chunks (only 20 used here)  
- `train_from_chunks.ipynb` — Full training notebook  
- `train_tokenizer.ipynb` — BPE tokenizer training  
- `test_custom_GPT2.ipynb` — Testing the model  
- `generate_text.py` — Script for text generation  


## 🚀 How to Run

To generate text with the trained model:

```bash
python generate_text.py --prompt "Once upon a time" --max_length 100  



