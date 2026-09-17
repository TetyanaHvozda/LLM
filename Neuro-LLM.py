import torch
import tiktoken
import urllib.request
import numpy as np
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

# Import architecture from Chapter 4
from chapter_4 import GPTModel 

# ==========================================
# 1. CONFIGURATION & DEVICE SETUP
# ==========================================
GPT_CONFIG_124M = {
    "vocab_size": 50257,   
    "context_length": 1024, # Set to 1024 for full texts
    "emb_dim": 768,        
    "n_heads": 12,         
    "n_layers": 12,        
    "drop_rate": 0.1,      
    "qkv_bias": True       # Must be True to match OpenAI's GPT-2 weights
}

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using {device} device.")

tokenizer = tiktoken.get_encoding("gpt2")

# ==========================================
# 2. HUGGING FACE DATA PIPELINE
# ==========================================
def is_neuro(example):
    try:
        front_text = " ".join(example.get("front", [])).lower()
        return any(kw in front_text for kw in [
            "electroencephalography", "eeg", "brain-computer", "neural decoding", "bci", "neurotechnology", "neuroscience", "neuroprosthetics", "neuroergonomics", "fMRI", "deep brain stimulation", "neuroplasticity"
        ])
    except TypeError:
        return False

def extract_and_tokenize(example):
    body_paragraphs = example.get("body", [])
    full_text = "\n\n".join(body_paragraphs)
    token_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
    token_ids.append(tokenizer.eot_token)
    return {"token_ids": token_ids}

class StreamPackingDataset(IterableDataset):
    def __init__(self, tokenized_stream, max_length):
        self.stream = tokenized_stream
        self.max_length = max_length

    def __iter__(self):
        buffer = []
        for example in self.stream:
            buffer.extend(example["token_ids"])
            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:] 
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

# Initialize the stream
dataset = load_dataset("TomTBT/pmc_open_access_xml", split="commercial", streaming=True)
neuro_stream = dataset.filter(is_neuro).map(extract_and_tokenize)
train_loader = DataLoader(StreamPackingDataset(neuro_stream, max_length=1024), batch_size=2)
# For a real run, you would create a separate validation stream. 
# Here we use the same structure for simplicity.
val_loader = DataLoader(StreamPackingDataset(neuro_stream, max_length=1024), batch_size=2) 

# ==========================================
# 3. UTILITY & GENERATION FUNCTIONS
# ==========================================
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# ==========================================
# 4. TRAINING LOOP MATH
# ==========================================
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

def calc_loss_loader(data_loader, model, device, num_batches=5):
    total_loss = 0.
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_model_simple(model, train_loader, val_loader, optimizer, device, max_steps, eval_freq, start_context):
    global_step = 0
    model.train()
    
    for input_batch, target_batch in train_loader:
        if global_step >= max_steps:
            break
            
        optimizer.zero_grad() 
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward() 
        optimizer.step() 
        global_step += 1

        if global_step % eval_freq == 0:
            train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter=5)
            print(f"Step {global_step:06d}: Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            
            # Print a sample generation to track progress
            model.eval()
            encoded = text_to_token_ids(start_context, tokenizer).to(device)
            with torch.no_grad():
                token_ids = generate(model, encoded, max_new_tokens=25, context_size=1024, top_k=10, temperature=0.1)
            print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
            model.train()

# ==========================================
# 5. PRE-TRAINED WEIGHT LOADING 
# ==========================================
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)
        
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

from gpt_download import download_and_load_gpt2

# ==========================================
# 6. EXECUTE PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Downloading and applying OpenAI GPT-2 weights...")
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    
    model = GPTModel(GPT_CONFIG_124M)
    load_weights_into_gpt(model, params)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1) # Lowered LR for fine-tuning
    
    print("Starting Domain-Adaptive Pretraining...")
    train_model_simple(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        device=device,
        max_steps=5000, 
        eval_freq=500, 
        start_context="The electroencephalogram (EEG) signal displays"
    )
    
    print("Saving customized Neuro-LLM...")
    torch.save(model.state_dict(), "neuro_model_weights.pth")