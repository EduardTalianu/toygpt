from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import time
import math

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import tiktoken

from model_mhc_expert import mHCExpertTransformer


# ============================================================
# File selection helpers
# ============================================================

def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    root.destroy()
    return Path(path) if path else None


# ============================================================
# Learning Rate Warmup Scheduler
# ============================================================

def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    """
    Learning rate schedule with linear warmup followed by cosine decay.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# Dataset iterator with train/val split
# ============================================================

def split_data(encoded, val_ratio=0.1):
    """Split data into train and validation sets."""
    n = len(encoded)
    split_idx = int(n * (1 - val_ratio))
    return encoded[:split_idx], encoded[split_idx:]


def text_batches(encoded, seq_len, batch_size, device, shuffle=True):
    max_pos = encoded.size(0) - seq_len - 1
    if max_pos <= 0:
        return
    
    if shuffle:
        indices = torch.randperm(max_pos)
    else:
        indices = torch.arange(max_pos)

    for i in range(0, len(indices), batch_size):
        idx = indices[i:i + batch_size]
        x = torch.stack([encoded[j:j + seq_len] for j in idx])
        y = torch.stack([encoded[j + 1:j + seq_len + 1] for j in idx])
        yield x.to(device), y.to(device)


# ============================================================
# Validation function
# ============================================================

@torch.no_grad()
def validate(model, val_data, seq_len, batch_size, device, dtype):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    num_batches = 0
    
    for x, y in text_batches(val_data, seq_len, batch_size, device, shuffle=False):
        with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype == torch.bfloat16)):
            logits, _, _, _ = model(x, use_cache=False, compute_entropy=False, update_router_mem=False)
            
            loss_ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
        
        total_loss += loss_ce.item()
        total_ce += loss_ce.item()
        num_batches += 1
        
        if num_batches >= 20:  # Limit validation batches for speed
            break
    
    model.train()
    return {
        'loss': total_loss / num_batches,
        'ce': total_ce / num_batches,
    }


# ============================================================
# Training with Warmup + Gradient Accumulation
# ============================================================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    print("="*60)
    print(" STABLE mHC Expert Training (v2)")
    print(" Architecture Fixes Applied")
    print("="*60)
    print(f" Device: {device}")
    print(f" Precision: {dtype} {'(2x memory savings!)' if use_bf16 else ''}")
    if device == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f" VRAM: {vram_gb:.1f} GB")
    print(" Architecture Features:")
    print("  ✅ Convex Combination (Spectral Norm <= 1)")
    print("  ✅ Sinkhorn in FP32 (Numerical Stability)")
    print("  ✅ Router Memory (Passive Tracking Only)")
    print("  ✅ Depth Gate (Consistent Application)")
    print("  ✅ SwiGLU experts")
    print("  ✅ KV cache for inference")
    print("="*60 + "\n")

    # ----------------------------
    # Select model file
    # ----------------------------
    model_path = select_file(
        title="Select STABLE mHC v2 model file",
        filetypes=[("PyTorch model", "*.pt")]
    )

    if model_path is None:
        messagebox.showerror("Error", "No model file selected.")
        return

    # ----------------------------
    # Select training text
    # ----------------------------
    text_path = select_file(
        title="Select training text (book)",
        filetypes=[("Text files", "*.txt")]
    )

    if text_path is None:
        messagebox.showerror("Error", "No training text selected.")
        return

    print(f"Model: {model_path}")
    print(f"Training text: {text_path}")

    # ----------------------------
    # Load text and tokenizer
    # ----------------------------
    text = text_path.read_text(encoding="utf-8")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = torch.tensor(tokenizer.encode(text, allowed_special="all"), dtype=torch.long)
    
    vocab_size = tokenizer.n_vocab

    print(f"Text length: {len(text):,} characters")
    print(f"Token count: {len(encoded):,} tokens")
    print(f"Vocab size: {vocab_size}")

    # Split into train/val
    train_data, val_data = split_data(encoded, val_ratio=0.1)
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")

    # ----------------------------
    # Load model
    # ----------------------------
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]

    config.pop("architecture", None)

    if config["vocab_size"] != vocab_size:
        messagebox.showerror(
            "Vocab mismatch",
            f"Model vocab size = {config['vocab_size']}\n"
            f"Tokenizer vocab size = {vocab_size}\n\n"
            "Recreate the model with matching vocab size."
        )
        return

    model = mHCExpertTransformer(**config).to(device)
    
    if use_bf16:
        model = model.to(dtype=dtype)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    n_expansion = config.get('n_expansion', 4)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Number of shared SwiGLU experts: {config.get('num_experts', 8)}")
    print(f"mHC stream expansion: {n_expansion}x")
    print(f"Architecture: STABLE mHC v2")
    
    param_memory = total_params * (2 if use_bf16 else 4) / 1e9
    print(f"Model memory: ~{param_memory:.2f} GB")

    # ----------------------------
    # Training Hyperparameters
    # ----------------------------
    
    total_training_steps = 50000
    warmup_steps = 2000 
    
    peak_lr = 3e-4
    min_lr = 1e-5
    
    optimizer = AdamW(
        model.parameters(), 
        lr=peak_lr,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    scheduler = get_warmup_cosine_scheduler(
        optimizer, 
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
        min_lr_ratio=min_lr/peak_lr
    )
    
    gradient_accumulation_steps = 4
    
    # Loss penalties
    lambda_lb = 0.005
    lambda_ent_moe = 0.005
    lambda_ent_attn = 0.002

    seq_len = 256
    batch_size = 8 
    
    num_epochs = 1  
    log_interval = 100
    val_interval = 1000
    checkpoint_interval = 500  
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    min_steps = 5000

    print("\n" + "="*60)
    print("✅ STABLE mHC TRAINING CONFIGURATION")
    print("="*60)
    print(f"Architecture: STABLE mHC v2")
    print(f"Spectral Norm Guarantee: ENABLED ✅ (Convex Combo)")
    print(f"Sinkhorn Precision: FP32 Internally ✅")
    print(f"Total target steps: {total_training_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"  -> Router memory is passive; updates enabled post-warmup")
    print(f"  -> to ensure accurate load statistics.")
    print(f"Peak LR: {peak_lr} -> Min LR: {min_lr}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps} steps ✅")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Sequence length: {seq_len}")
    print(f"Weight decay: 0.1")
    print(f"Betas: (0.9, 0.95)")
    print(f"Load balance penalty: {lambda_lb}")
    print(f"MoE Entropy penalty: {lambda_ent_moe}")
    print(f"Attn Entropy penalty: {lambda_ent_attn}")
    print(f"Gradient clip: 1.0")
    print(f"Min steps before early stop: {min_steps}")
    print(f"Patience: {patience}")
    print(f"Mixed precision: {use_bf16}")
    print("="*60 + "\n")

    print("Starting STABLE mHC training...\n")

    start_time = time.time()

    global_step = 0
    accumulation_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*60}\n")
        
        epoch_steps = 0
        
        for x, y in text_batches(train_data, seq_len, batch_size, device):
            with torch.autocast(device_type=device, dtype=dtype, enabled=use_bf16):
                is_last_accum_step = (accumulation_step == gradient_accumulation_steps - 1)
                in_warmup = global_step < warmup_steps
                # Memory logic: Still disable update during warmup to avoid noisy stats, 
                # but it's no longer critical for stability.
                should_update_router_mem = is_last_accum_step and not in_warmup
                
                logits, lb_loss, ent_attn, ent_moe = model(
                    x, use_cache=False, compute_entropy=True, 
                    update_router_mem=should_update_router_mem
                )

                loss_ce = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

                loss = loss_ce + lambda_lb * lb_loss + (lambda_ent_moe * ent_moe) + (lambda_ent_attn * ent_attn)
                loss = loss / gradient_accumulation_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ NaN/Inf detected at step {global_step}!")
                print(f"   CE: {loss_ce.item()}, LB: {lb_loss.item()}, E_MoE: {ent_moe.item()}, E_Att: {ent_attn.item()}")
                print("   Skipping this batch and continuing...")
                global_step += 1
                epoch_steps += 1
                accumulation_step = 0
                optimizer.zero_grad()
                continue

            loss.backward()
            
            accumulation_step += 1
            
            if accumulation_step >= gradient_accumulation_steps:
                nan_grads = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"⚠️ NaN/Inf gradient in {name}")
                            nan_grads = True
                            break
                
                if nan_grads:
                    print("   Skipping optimizer step due to bad gradients")
                    optimizer.zero_grad()
                    accumulation_step = 0
                    global_step += 1
                    epoch_steps += 1
                    continue
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                accumulation_step = 0
                global_step += 1
                epoch_steps += 1
                
                if global_step == warmup_steps:
                    print(f"\n{'='*60}")
                    print(f"✅ WARMUP COMPLETE")
                    print(f"   LR has reached {optimizer.param_groups[0]['lr']:.2e}")
                    print(f"   Router Load Tracking now active.")
                    print(f"{'='*60}\n")
                
                if global_step > 0 and global_step % checkpoint_interval == 0:
                    checkpoint_path = model_path.with_name(f"{model_path.stem}_checkpoint.pt")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "tokenizer_name": "gpt2",
                            "step": global_step,
                            "dtype": str(dtype),
                            "architecture": "mhc_stable_v2",
                            "version": "stable_mhc_v2"
                        },
                        checkpoint_path,
                    )
                    print(f"💾 Checkpoint saved: {checkpoint_path.name}")

                if global_step % log_interval == 0:
                    elapsed_time = time.time() - start_time
                    if global_step > 0 and elapsed_time > 0:
                        steps_per_sec = global_step / elapsed_time
                        remaining_steps = total_training_steps - global_step
                        remaining_time_sec = remaining_steps / steps_per_sec
                        
                        hrs = int(remaining_time_sec // 3600)
                        mins = int((remaining_time_sec % 3600) // 60)
                        secs = int(remaining_time_sec % 60)
                        eta_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
                    else:
                        remaining_steps = total_training_steps
                        eta_str = "--:--:--"

                    current_lr = optimizer.param_groups[0]['lr']
                    
                    actual_loss = loss.item() * gradient_accumulation_steps
                    loss_str = f"{actual_loss:.4f}" if not torch.isnan(loss) else "NaN"
                    ce_str = f"{loss_ce.item():.4f}" if not torch.isnan(loss_ce) else "NaN"
                    lb_str = f"{lb_loss.item():.4f}" if not torch.isnan(lb_loss) else "NaN"
                    ent_m_str = f"{ent_moe.item():.4f}" if not torch.isnan(ent_moe) else "NaN"
                    ent_a_str = f"{ent_attn.item():.4f}" if not torch.isnan(ent_attn) else "NaN"
                    
                    diversity_indicator = ""
                    if lb_loss.item() > 50.0:
                        diversity_indicator = " ⚠️COLLAPSE"
                    elif lb_loss.item() > 20.0:
                        diversity_indicator = " ⚠️LOW_DIV"
                    
                    if global_step < warmup_steps:
                        warmup_pct = (global_step / warmup_steps) * 100
                        warmup_indicator = f" [WARMUP {warmup_pct:5.1f}%]"
                    else:
                        warmup_indicator = ""
                    
                    print(
                        f"step {global_step:5d}{warmup_indicator} | "
                        f"loss {loss_str:>8s} | "
                        f"CE {ce_str:>8s} | "
                        f"LB {lb_str:>8s}{diversity_indicator} | "
                        f"E_M {ent_m_str:>8s} | "
                        f"E_A {ent_a_str:>8s} | "
                        f"LR {current_lr:.2e} | "
                        f"Rem: {remaining_steps:5d} | "
                        f"ETA: {eta_str}"
                    )
                
                if global_step % val_interval == 0 and global_step > 0:
                    val_metrics = validate(model, val_data, seq_len, batch_size, device, dtype)
                    
                    print(f"\n{'*'*60}")
                    print(f"VALIDATION @ step {global_step}")
                    print(f"Val Loss: {val_metrics['loss']:.4f} | "
                          f"Val CE: {val_metrics['ce']:.4f}")
                    print(f"{'*'*60}\n")
                    
                    if global_step >= min_steps:
                        if val_metrics['loss'] < best_val_loss:
                            best_val_loss = val_metrics['loss']
                            patience_counter = 0
                            
                            best_path = model_path.with_name(model_path.stem + "_best.pt")
                            torch.save(
                                {
                                    "model_state_dict": model.state_dict(),
                                    "config": config,
                                    "tokenizer_name": "gpt2",
                                    "val_loss": best_val_loss,
                                    "dtype": str(dtype),
                                    "architecture": "mhc_stable_v2",
                                    "version": "stable_mhc_v2"
                                },
                                best_path,
                            )
                            print(f"✓ New best model saved! Val loss: {best_val_loss:.4f}\n")
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"\n⚠️ RECOMMENDATION: Consider stopping training. "
                                      f"Val loss hasn't improved for {patience_counter} checks.\n")
                            else:
                                print(f"⏸ Patience: {patience_counter}/{patience}\n")
                    else:
                        if val_metrics['loss'] < best_val_loss:
                            best_val_loss = val_metrics['loss']
                            best_path = model_path.with_name(model_path.stem + "_best.pt")
                            torch.save(
                                {
                                    "model_state_dict": model.state_dict(),
                                    "config": config,
                                    "tokenizer_name": "gpt2",
                                    "val_loss": best_val_loss,
                                    "dtype": str(dtype),
                                    "architecture": "mhc_stable_v2",
                                    "version": "stable_mhc_v2"
                                },
                                best_path,
                            )
                            print(f"✓ New best model saved! Val loss: {best_val_loss:.4f}\n")
                        print(f"ℹ Early stop check disabled until step {min_steps}\n")
        
        print(f"\nEpoch {epoch + 1} complete: {epoch_steps} steps\n")

    output_path = model_path.with_name(model_path.stem + "_trained.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "tokenizer_name": "gpt2",
            "final_step": global_step,
            "dtype": str(dtype),
            "architecture": "mhc_stable_v2",
            "version": "stable_mhc_v2"
        },
        output_path,
    )

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Architecture: STABLE mHC v2")
    print(f"Total steps: {global_step}")
    print(f"Final model: {output_path}")
    if best_val_loss < float('inf'):
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Best model: {model_path.with_name(model_path.stem + '_best.pt')}")
    print(f"Precision: {dtype}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print("="*60)

    messagebox.showinfo(
        "Training complete",
        f"Trained STABLE mHC v2 model saved to:\n{output_path}\n\n"
        f"Total steps: {global_step}\n"
        f"Best val loss: {best_val_loss:.4f}\n"
        f"Architecture: STABLE mHC v2\n"
        f"Features:\n"
        f"✅ Spectral Norm Guarantee (Convex Combo)\n"
        f"✅ Sinkhorn FP32 Stability\n"
        f"✅ Clean Router Logic\n"
        f"✅ LR warmup ({warmup_steps} steps)\n"
        f"✅ Gradient accumulation (4x)"
    )


if __name__ == "__main__":
    train()