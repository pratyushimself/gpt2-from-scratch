import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# === Step 1: Load tokenizer ===
tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_gpt2_custom")

# === Step 2: Load trained model ===
model_path = "gpt2_custom_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# === Step 3: Move to CPU (or CUDA if available) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Step 4: Generate text from a prompt ===
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Adjust these settings as needed
max_length = 100
num_return_sequences = 1
temperature = 0.9
top_k = 50

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

# === Step 5: Decode and print result ===
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n📝 Generated Text:\n")
print(generated_text)
