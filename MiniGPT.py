import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use other GPT-2 variants as well
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate text
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=50256)

for generated_sequence in output:
    generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    print(generated_text)
