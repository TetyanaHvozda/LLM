import chainlit as cl
import torch
import tiktoken
from chapter_4 import GPTModel

# Load the same config we used for training
GPT_CONFIG_124M = {
    "vocab_size": 50257, "context_length": 1024, "emb_dim": 768,
    "n_heads": 12, "n_layers": 12, "drop_rate": 0.0, "qkv_bias": True
}

@cl.on_chat_start
async def on_chat_start():
    # 1. Initialize architecture
    model = GPTModel(GPT_CONFIG_124M)
    
    # 2. Load trained weights
    model.load_state_dict(torch.load("neuro_model_weights.pth", map_location="cpu"))
    model.eval()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Store them in the user session so they are available for messages
    cl.user_session.set("model", model)
    cl.user_session.set("tokenizer", tokenizer)
    
    await cl.Message(content="Neuro-LLM initialized. Ask me about computational neuroscience.").send()

@cl.on_message
async def on_message(message: cl.Message):
    model = cl.user_session.get("model")
    tokenizer = cl.user_session.get("tokenizer")
    
    # Convert user message to tokens
    input_ids = tokenizer.encode(message.content, allowed_special={'<|endoftext|>'})
    idx = torch.tensor(input_ids).unsqueeze(0)
    
    # Generate the response (using a simplified generation loop for the UI)
    context_size = 1024
    for _ in range(50): # Max new tokens
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus on the last time step and get the most likely token
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next.item() == tokenizer.eot_token:
            break
            
        idx = torch.cat((idx, idx_next), dim=1)
    
    # Decode and send the response back to the UI
    output_text = tokenizer.decode(idx.squeeze(0).tolist())
    
    # Strip the original prompt from the output so it only prints the answer
    final_answer = output_text[len(message.content):].strip()
    
    await cl.Message(content=final_answer).send()