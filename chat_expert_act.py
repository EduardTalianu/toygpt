import torch
import torch.nn.functional as F
from pathlib import Path
import tiktoken
import math

from model_mhc_expert import mHCExpertTransformer


# ============================================================
# Entropy-based confidence calculation
# ============================================================

def calculate_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum()
    max_prob = probs.max().item()
    max_entropy = math.log(logits.size(-1))
    normalized_entropy = (entropy / max_entropy).item()
    confidence = 1.0 - normalized_entropy
    return entropy.item(), confidence, max_prob


def adaptive_temperature(entropy, base_temp=0.8, min_temp=0.3, max_temp=1.5, entropy_scale=2.0):
    scaled_entropy = min(1.0, entropy * entropy_scale)
    temp = min_temp + (max_temp - min_temp) * scaled_entropy
    return temp


# ============================================================
# Sampling helpers
# ============================================================

def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ============================================================
# Text generation with KV Cache
# ============================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=300,
    base_temperature=0.8,
    top_k=50,
    top_p=0.95,
    device="cpu",
    dtype=torch.float32,
    use_dynamic_temp=True,
    min_temp=0.3,
    max_temp=1.5,
):
    """
    ✅ OPTIMIZED GENERATION for STABLE mHC v2
    """
    model.eval()
    prompt_tokens = tokenizer.encode(prompt)
    
    vocab_size = model.vocab_size
    prompt_tokens = [t for t in prompt_tokens if 0 <= t < vocab_size]
    
    if not prompt_tokens:
        print("Error: No valid tokens in prompt")
        return
    
    ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    kv_caches = None
    generated_count = 0
    
    for i in range(max_new_tokens):
        if ids.size(1) >= model.max_seq_len:
            ids = ids[:, -model.max_seq_len + 1:]
            kv_caches = None
        
        try:
            with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype == torch.bfloat16)):
                if i == 0:
                    input_ids = ids
                else:
                    input_ids = ids[:, -1:]
                
                logits, _, _, _, kv_caches = model(
                    input_ids,
                    kv_caches=kv_caches,
                    use_cache=True,
                    compute_entropy=False, 
                    update_router_mem=False
                )
                
                next_logits = logits[0, -1]
            
            if torch.isnan(next_logits).any() or torch.isinf(next_logits).any():
                print("\nWarning: NaN/Inf in logits, stopping generation")
                break

            entropy, confidence, max_prob = calculate_entropy(next_logits)
            
            if use_dynamic_temp:
                current_temp = adaptive_temperature(
                    entropy,
                    base_temp=base_temperature,
                    min_temp=min_temp,
                    max_temp=max_temp,
                    entropy_scale=2.0
                )
            else:
                current_temp = base_temperature

            next_id = sample_next(next_logits, temperature=current_temp, top_k=top_k, top_p=top_p)
            
            if next_id < 0 or next_id >= vocab_size:
                print(f"\nWarning: Invalid token ID {next_id}, stopping generation")
                break
            
            next_token_text = tokenizer.decode([next_id])
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
            
            if next_id == tokenizer.eot_token:
                break
            
            generated_count += 1
            
            yield {
                'token': next_token_text,
                'confidence': confidence,
                'max_prob': max_prob,
                'entropy': entropy,
                'temperature': current_temp
            }
        except Exception as e:
            print(f"\nError during generation step {i}: {e}")
            break


# ============================================================
# Chat loop
# ============================================================

def chat():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    print("="*60)
    print(" ✅ STABLE mHC v2 Chat")
    print("="*60)
    print(f" Device: {device}")
    print(f" Precision: {dtype}")
    if device == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(" Architecture Features:")
    print("  ✅ Convex Combination (Stable)")
    print("  ✅ Sinkhorn FP32 (Accurate)")
    print("  ✅ KV cache (Fast)")
    print("  ✅ SwiGLU experts")
    print("="*60 + "\n")

    # Priority loading for Stable v2 files
    model_path = Path("mhc_stable_transformer_checkpoint.pt")
    if not model_path.exists():
        model_path = Path("mhc_stable_transformer_best.pt")
    if not model_path.exists():
        model_path = Path("mhc_stable_transformer_trained.pt")
    if not model_path.exists():
        model_path = Path("mhc_stable_transformer.pt")
    
    if not model_path.exists():
        print("ERROR: No trained STABLE mHC model found.")
        print("Expected: mhc_stable_transformer_*.pt")
        return

    print(f"Loading model: {model_path.name}")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    
    config.pop("architecture", None)

    tokenizer_name = checkpoint.get("tokenizer_name", "gpt2")
    tokenizer = tiktoken.get_encoding(tokenizer_name)

    model = mHCExpertTransformer(**config).to(device)
    
    if use_bf16:
        model = model.to(dtype=dtype)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    num_experts = config.get("num_experts", 8)
    n_expansion = config.get("n_expansion", 4)

    print("="*60)
    print(" Model Information")
    print("="*60)
    print(f" Architecture: STABLE mHC v2")
    print(f" Features: Convex Combo, FP32 Sinkhorn, KV Cache")
    print(f" Model: {model_path.name}")
    print(f" Tokenizer: {tokenizer_name}")
    print(f" Vocab size: {tokenizer.n_vocab:,}")
    print(f" Parameters: {total_params:,}")
    print(f" Shared SwiGLU Experts: {num_experts}")
    print(f" Stream expansion: {n_expansion}x")
    print(f" Precision: {dtype}")
    if "val_loss" in checkpoint:
        print(f" Val loss: {checkpoint['val_loss']:.4f}")
    if "version" in checkpoint:
        print(f" Version: {checkpoint['version']}")
    param_memory = total_params * (2 if use_bf16 else 4) / 1e9
    print(f" Model size: ~{param_memory:.2f} GB")
    print("="*60)
    print("\nCommands:")
    print("  /dynamic         - Enable dynamic temperature (default)")
    print("  /fixed           - Use fixed temperature")
    print("  /temp <value>    - Set base temperature (default: 0.8)")
    print("  /stats           - Toggle generation statistics")
    print("  /verbose         - Toggle per-token stats")
    print("  quit/exit        - Exit chat")
    print("="*60 + "\n")

    base_temperature = 0.8
    min_temp = 0.3
    max_temp = 1.5
    use_dynamic_temp = True
    show_stats = False
    show_verbose = False

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if user_input.startswith("/dynamic"):
            use_dynamic_temp = True
            print(f"✓ Dynamic temperature enabled (range: {min_temp}-{max_temp})\n")
            continue
        
        if user_input.startswith("/fixed"):
            use_dynamic_temp = False
            print(f"✓ Fixed temperature enabled (temp: {base_temperature})\n")
            continue
        
        if user_input.startswith("/temp "):
            try:
                base_temperature = float(user_input.split()[-1])
                if base_temperature <= 0:
                    print("⚠️ Temperature must be positive\n")
                    continue
                print(f"✓ Base temperature set to {base_temperature}\n")
                continue
            except (ValueError, IndexError):
                print("Usage: /temp <value>\n")
                continue
        
        if user_input.startswith("/stats"):
            show_stats = not show_stats
            print(f"✓ Statistics {'enabled' if show_stats else 'disabled'}\n")
            continue
        
        if user_input.startswith("/verbose"):
            show_verbose = not show_verbose
            print(f"✓ Verbose mode {'enabled' if show_verbose else 'disabled'}\n")
            continue

        prompt = user_input + "\n"

        try:
            total_confidence = 0.0
            total_entropy = 0.0
            min_confidence = 1.0
            max_confidence = 0.0
            generated_tokens = 0
            
            print("\nModel: ", end="", flush=True)
            
            for result in generate(
                model, tokenizer, prompt, max_new_tokens=400,
                base_temperature=base_temperature, top_k=50, top_p=0.95,
                device=device, dtype=dtype,
                use_dynamic_temp=use_dynamic_temp, min_temp=min_temp, max_temp=max_temp
            ):
                print(result['token'], end="", flush=True)
                total_confidence += result['confidence']
                total_entropy += result['entropy']
                min_confidence = min(min_confidence, result['confidence'])
                max_confidence = max(max_confidence, result['confidence'])
                generated_tokens += 1
                
                if show_verbose:
                    print(f" [C:{result['confidence']:.2f} T:{result['temperature']:.2f}]", end="", flush=True)
            
            print()
            avg_confidence = total_confidence / max(1, generated_tokens)
            avg_entropy = total_entropy / max(1, generated_tokens)
            
            if show_stats:
                print(f"\n{'='*60}")
                print(f"Generation Statistics:")
                print(f"  Tokens: {generated_tokens}")
                print(f"  Avg confidence: {avg_confidence:.2f}")
                print(f"  Confidence range: {min_confidence:.2f} - {max_confidence:.2f}")
                print(f"  Avg entropy: {avg_entropy:.3f}")
                print(f"  Dynamic temp: {'ON' if use_dynamic_temp else 'OFF'}")
                if use_dynamic_temp:
                    print(f"  Temp range: {min_temp:.1f} - {max_temp:.1f}")
                else:
                    print(f"  Fixed temp: {base_temperature:.1f}")
                print(f"  Architecture: STABLE mHC v2")
                print(f"  Convex Combination: ENABLED ✅")
                print(f"  KV cache: ENABLED ✅")
                print(f"{'='*60}")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"\nError during generation: {e}")
            import traceback
            traceback.print_exc()
            print("Try with a shorter prompt or different settings.\n")


if __name__ == "__main__":
    chat()