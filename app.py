import chainlit as cl
import torch
import tiktoken
from chapter_4 import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True
}

def format_prompt(instruction_text):
    return (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction_text}\n\n"
        f"### Response:\n"
    )

def sample_next_token(logits, temperature=0.7, top_k=30):
    logits = logits / temperature
    top_logits, _ = torch.topk(logits, top_k)
    min_val = top_logits[:, -1]
    logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

@cl.on_chat_start
async def on_chat_start():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("neuro_model_instruct_weights.pth", map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    cl.user_session.set("model", model)
    cl.user_session.set("tokenizer", tokenizer)
    cl.user_session.set("device", device)
    
    await cl.Message(content="Neuro-LLM initialized. Ask me about computational neuroscience.").send()

@cl.on_message
async def on_message(message: cl.Message):
    model = cl.user_session.get("model")
    tokenizer = cl.user_session.get("tokenizer")
    device = cl.user_session.get("device")
    
    full_prompt = format_prompt(message.content)
    input_ids = tokenizer.encode(full_prompt, allowed_special={'<|endoftext|>'})
    idx = torch.tensor(input_ids, device=device).unsqueeze(0)
    
    generated_tokens = []
    context_size = GPT_CONFIG_124M["context_length"]
    
    for _ in range(120):  # Maximum new tokens to generate
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        next_token = sample_next_token(logits[:, -1, :], temperature=0.7, top_k=30)
        
        if next_token.item() == tokenizer.eot_token:
            break
            
        generated_tokens.append(next_token.item())
        idx = torch.cat((idx, next_token), dim=1)
        
    response = tokenizer.decode(generated_tokens).strip()
    
    # Clean up any trailing template artifacts if generated
    if "### Instruction:" in response:
        response = response.split("### Instruction:")[0].strip()
        
    await cl.Message(content=response if response else "Unable to generate a response.").send()