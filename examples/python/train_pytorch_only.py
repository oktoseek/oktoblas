"""
PyTorch Training Benchmark (No OktoBLAS)
========================================
Training with PyTorch only - baseline comparison

pip install torch transformers datasets

Author: OktoSeek AI
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datetime import datetime

print("=" * 70)
print("📊 PYTORCH ONLY - Testing without OktoBLAS")
print("=" * 70)

# Configuration
CONFIG = {
    "model_name": "gpt2",
    "dataset_path": "D:/model_trainee/sharegpt_chat.jsonl",
    "max_examples": 10000,
    "max_length": 128,
    "batch_size": 8,
    "epochs": 1,
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "log_every": 10,
    "eval_every": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different formats
        if "chat" in item:
            # ShareGPT format: [{"role": "user", "content": "..."}, ...]
            chat = item["chat"]
            text = " ".join([c.get("content", "")[:200] for c in chat[:2]])
        elif "conversations" in item:
            text = " ".join([c.get("value", "") for c in item["conversations"][:2]])
        elif "text" in item:
            text = item["text"]
        elif "instruction" in item and "output" in item:
            text = f"{item['instruction']} {item['output']}"
        elif "question" in item and "response" in item:
            text = f"{item['question']} {item['response']}"
        else:
            text = str(item)[:500]
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

def load_dataset(path, max_examples):
    """Load JSONL dataset"""
    data = []
    print(f"\n📂 Loading dataset from {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            try:
                data.append(json.loads(line))
            except:
                continue
    
    print(f"✅ Loaded {len(data)} examples")
    return data

def format_time(seconds):
    """Format seconds to human readable"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def train():
    print("\n" + "=" * 70)
    print("📊 TRAINING WITH PYTORCH ONLY (BASELINE)")
    print("=" * 70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Device: {CONFIG['device']}")
    print(f"Examples: {CONFIG['max_examples']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Max length: {CONFIG['max_length']}")
    print("=" * 70)
    
    # Load tokenizer and model
    print("\n📦 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"])
    model.to(CONFIG["device"])
    model.train()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model loaded: {total_params/1e6:.1f}M parameters ({trainable_params/1e6:.1f}M trainable)")
    
    # Load dataset
    data = load_dataset(CONFIG["dataset_path"], CONFIG["max_examples"])
    dataset = ChatDataset(data, tokenizer, CONFIG["max_length"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    total_steps = len(dataloader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, CONFIG["warmup_steps"], total_steps)
    
    # Training metrics
    global_step = 0
    total_loss = 0
    start_time = time.time()
    step_times = []
    losses = []
    
    print(f"\n🏋️ Starting training... ({len(dataloader)} batches per epoch)")
    print("-" * 70)
    
    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()
            
            # Move to device
            input_ids = batch["input_ids"].to(CONFIG["device"])
            attention_mask = batch["attention_mask"].to(CONFIG["device"])
            labels = batch["labels"].to(CONFIG["device"])
            
            # Forward pass (PyTorch only)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Metrics
            step_time = time.time() - step_start
            step_times.append(step_time)
            total_loss += loss.item()
            epoch_loss += loss.item()
            losses.append(loss.item())
            global_step += 1
            
            # Calculate speed
            examples_per_sec = CONFIG["batch_size"] / step_time
            
            # Log
            if global_step % CONFIG["log_every"] == 0:
                avg_loss = total_loss / global_step
                avg_step_time = sum(step_times[-100:]) / len(step_times[-100:])
                eta_seconds = avg_step_time * (total_steps - global_step)
                
                # Calculate approximate TFLOPS (for GPT-2 small)
                flops_per_step = 6 * total_params * CONFIG["batch_size"] * CONFIG["max_length"]
                tflops = flops_per_step / step_time / 1e12
                
                print(f"[PyTorch] Step {global_step:5d}/{total_steps} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Avg: {avg_loss:.4f} | "
                      f"Speed: {examples_per_sec:.1f} ex/s | "
                      f"TFLOPS: {tflops:.2f} | "
                      f"ETA: {format_time(eta_seconds)}")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss / len(dataloader)
        epoch_speed = len(dataset) / epoch_time
        
        print("-" * 70)
        print(f"📊 Epoch {epoch+1}/{CONFIG['epochs']} Complete")
        print(f"   Loss: {epoch_avg_loss:.4f}")
        print(f"   Time: {format_time(epoch_time)}")
        print(f"   Speed: {epoch_speed:.1f} examples/sec")
        print("-" * 70)
    
    # Final summary
    total_time = time.time() - start_time
    final_avg_loss = total_loss / global_step
    overall_speed = CONFIG["max_examples"] / total_time
    
    print("\n" + "=" * 70)
    print("🏆 TRAINING COMPLETE - PYTORCH ONLY (BASELINE)")
    print("=" * 70)
    print(f"Total time: {format_time(total_time)}")
    print(f"Final loss: {final_avg_loss:.4f}")
    print(f"Average speed: {overall_speed:.1f} examples/sec")
    print(f"Total steps: {global_step}")
    
    # Save results
    results = {
        "backend": "pytorch",
        "model": CONFIG["model_name"],
        "examples": CONFIG["max_examples"],
        "batch_size": CONFIG["batch_size"],
        "total_time_seconds": total_time,
        "final_loss": final_avg_loss,
        "examples_per_second": overall_speed,
        "total_steps": global_step,
        "timestamp": datetime.now().isoformat()
    }
    
    result_file = "training_result_pytorch.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to {result_file}")
    
    return results

if __name__ == "__main__":
    results = train()

