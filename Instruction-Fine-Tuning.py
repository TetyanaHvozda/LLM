import json
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from chapter_4 import GPTModel
from chapter_5 import train_model_simple, generate, text_to_token_ids, token_ids_to_text

# ==========================================
# 1. LOAD YOUR CUSTOM DATASET
# ==========================================
# Assumes a JSON format: [{"instruction": "...", "input": "...", "output": "..."}]
with open("neuro-instruction-data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text

# Partitioning
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# ==========================================
# 2. DATASET & BATCHING (Unchanged from book)
# ==========================================
tokenizer = tiktoken.get_encoding("gpt2")

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            full_text = format_input(entry) + f"\n\n### Response:\n{entry['output']}"
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index): return self.encoded_texts[index]
    def __len__(self): return len(self.data)

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=1024, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy() + [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Mask padding tokens so the model isn't penalized for them
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

train_loader = DataLoader(InstructionDataset(train_data, tokenizer), batch_size=8, collate_fn=lambda b: custom_collate_fn(b, device=device), shuffle=True)
val_loader = DataLoader(InstructionDataset(val_data, tokenizer), batch_size=8, collate_fn=lambda b: custom_collate_fn(b, device=device), shuffle=False)

# ==========================================
# 3. LOAD NEURO-LLM WEIGHTS
# ==========================================
GPT_CONFIG_124M = {
    "vocab_size": 50257, "context_length": 1024, "emb_dim": 768,
    "n_heads": 12, "n_layers": 12, "drop_rate": 0.1, "qkv_bias": True
}

model = GPTModel(GPT_CONFIG_124M)

# CRITICAL: Load the Domain-Adaptive weights, NOT OpenAI's base weights
model.load_state_dict(torch.load("neuro_model_weights.pth", map_location=device))
model.to(device)

# ==========================================
# 4. INSTRUCTION FINE-TUNING
# ==========================================
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002, weight_decay=0.2)

print("Starting Instruction Fine-Tuning...")
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=5, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

torch.save(model.state_dict(), "neuro_model_instruct_weights.pth")
print("Instruction Fine-Tuning Complete. Model saved.")