import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import tiktoken
import math

print("STARTING SCRIPT: model_mhc_expert.py (STABLE v2 with Fixed Spectral Norm & FP32 Sinkhorn).")

# ============================================================
# Utility: Entropy Calculation
# ============================================================

def entropy(p, eps=1e-8):
    """
    Computes Shannon entropy of a probability distribution tensor.
    """
    p = p.clamp(min=eps)
    return -(p * torch.log(p)).sum(dim=-1)

# ============================================================
# ✅ SINKHORN-KNOPP ALGORITHM (Fixed for Numerical Stability)
# ============================================================

def sinkhorn_knopp(H, num_iter=20, eps=1e-8):
    """
    Sinkhorn-Knopp algorithm to project matrix onto doubly stochastic manifold.
    
    FIX 2: Enforces float32 computation to prevent mass loss in bfloat16.
    """
    # Store original dtype to restore later
    input_dtype = H.dtype
    device = H.device
    
    # Cast to float32 for stable division operations
    H_float = H.float() if input_dtype != torch.float32 else H
    
    # Step 1: Make all entries positive via exp
    M = torch.exp(H_float)
    
    # Step 2: Iterative row and column normalization
    for _ in range(num_iter):
        # Row normalization: each row sums to 1
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        
        # Column normalization: each column sums to 1
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    
    # Cast back to original dtype (e.g., bfloat16)
    return M.to(input_dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Clamping for stability
        x = torch.clamp(x, min=-1e4, max=1e4)
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


# ============================================================
# RoPE (Rotary Position Embedding)
# ============================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len, device, dtype):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            self._cos_cached = freqs.cos()
            self._sin_cached = freqs.sin()
    
    def forward(self, x, seq_len):
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]
        return apply_rotary_pos_emb(x, cos, sin)


def apply_rotary_pos_emb(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


# ============================================================
# Module: Depth Gate
# ============================================================

class DepthGate(nn.Module):
    """Soft depth gating: scales the residual contribution of a layer."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, -2.0)

    def forward(self, x):
        gate = torch.sigmoid(self.proj(x))  # (B*T, 1)
        return gate


# ============================================================
# SwiGLU Expert
# ============================================================

class SwiGLUExpert(nn.Module):
    """
    SwiGLU activation - proven superior to GELU in modern LLMs.
    """
    def __init__(self, hidden_dim, expansion_factor=4):
        super().__init__()
        intermediate_dim = int(hidden_dim * expansion_factor * 2 / 3)
        
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.1)
        
        for layer in [self.w1, self.w2, self.w3]:
            nn.init.normal_(layer.weight, std=0.02)

    def forward(self, x):
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


# ============================================================
# ✅ STABLE mHC Connection Module (Fixed Spectral Norm & Sinkhorn)
# ============================================================

class mHCConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC) - STABLE VERSION
    
    Fixes:
    1. Sinkhorn-Knopp runs in float32.
    2. Global convex combination ensures Spectral Norm <= 1.
    """
    def __init__(self, hidden_dim, n_expansion=4, alpha_init=0.01, sinkhorn_iter=20):
        super().__init__()
        self.n = n_expansion
        self.hidden_dim = hidden_dim
        self.sinkhorn_iter = sinkhorn_iter
        
        # Linear projections for dynamic mappings
        self.phi_pre = nn.Linear(n_expansion * hidden_dim, n_expansion, bias=False)
        self.phi_post = nn.Linear(n_expansion * hidden_dim, n_expansion, bias=False)
        self.phi_res = nn.Linear(n_expansion * hidden_dim, n_expansion * n_expansion, bias=False)
        
        nn.init.normal_(self.phi_pre.weight, std=0.02)
        nn.init.normal_(self.phi_post.weight, std=0.02)
        nn.init.normal_(self.phi_res.weight, std=0.02)
        
        # Static biases
        self.bias_pre = nn.Parameter(torch.zeros(1, n_expansion))
        self.bias_post = nn.Parameter(torch.zeros(1, n_expansion))
        self.bias_res = nn.Parameter(torch.zeros(1, n_expansion, n_expansion))
        
        # Gating factors (initialized small)
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))
        
        # FIX 1: Stability Alpha for Convex Combination
        # Initialized to 0.5 (equal mix), in (0,1) via sigmoid
        self.stability_logit = nn.Parameter(torch.tensor(0.0))
        
        self.norm = RMSNorm(n_expansion * hidden_dim)
    
    def compute_mappings(self, x_stream, compute_entropy=True):
        B, T, N, D = x_stream.shape
        x_flat = x_stream.view(B * T, N * D)
        
        if torch.isnan(x_flat).any() or torch.isinf(x_flat).any():
            x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=1e4, neginf=-1e4)
        
        x_norm = self.norm(x_flat)
        
        # Dynamic Mappings
        h_pre_dynamic = self.phi_pre(x_norm)
        h_post_dynamic = self.phi_post(x_norm)
        h_res_dynamic = self.phi_res(x_norm).view(B * T, N, N)
        
        # Gating & Static Bias
        alpha_pre_clamped = torch.clamp(self.alpha_pre, min=0.0, max=0.1)
        alpha_post_clamped = torch.clamp(self.alpha_post, min=0.0, max=0.1)
        alpha_res_clamped = torch.clamp(self.alpha_res, min=0.0, max=0.1)
        
        h_pre_gated = alpha_pre_clamped * h_pre_dynamic + self.bias_pre
        h_post_gated = alpha_post_clamped * h_post_dynamic + self.bias_post
        h_res_gated = alpha_res_clamped * h_res_dynamic + self.bias_res
        
        # Manifold Projections
        H_pre = torch.sigmoid(torch.clamp(h_pre_gated, min=-10.0, max=10.0))
        H_post = 2.0 * torch.sigmoid(torch.clamp(h_post_gated, min=-10.0, max=10.0))
        
        # FIX 2: Sinkhorn called here (handles float32 conversion internally)
        H_res = sinkhorn_knopp(h_res_gated, num_iter=self.sinkhorn_iter)
        
        entropy_pre = 0.0
        entropy_post = 0.0
        if compute_entropy and self.training:
            H_pre_norm = H_pre / (H_pre.sum(dim=-1, keepdim=True) + 1e-8)
            H_post_norm = H_post / (H_post.sum(dim=-1, keepdim=True) + 1e-8)
            entropy_pre = entropy(H_pre_norm).mean()
            entropy_post = entropy(H_post_norm).mean()
        
        return H_pre, H_post, H_res, entropy_pre, entropy_post
    
    def apply_pre(self, x_stream, H_pre):
        """Collapse n streams → 1 using learned weights"""
        B, T, N, D = x_stream.shape
        x_flat = x_stream.view(B * T, N, D)
        H_pre_norm = H_pre / (H_pre.sum(dim=-1, keepdim=True) + 1e-8)
        x_single = torch.bmm(H_pre_norm.unsqueeze(1), x_flat).squeeze(1)
        return x_single
    
    def apply_post_and_res(self, x_stream, layer_output, H_post, H_res, gate=None):
        """
        Apply residual mapping and broadcast output back to streams.
        
        FIX 1: Spectral Norm <= 1 via Convex Combination.
        x_next = alpha * (H_res @ x) + (1 - alpha) * write_back
        """
        B, T, N, D = x_stream.shape
        x_flat = x_stream.view(B * T, N, D)
        
        # Residual mixing (Doubly Stochastic)
        mixed_stream = torch.bmm(H_res, x_flat)
        
        # Write Back calculation
        # Apply gate to H_post if provided (Dynamic Depth Gating)
        if gate is not None:
            H_post = H_post * gate
            
        H_post_norm = H_post / (H_post.sum(dim=-1, keepdim=True) + 1e-8)
        write_back = torch.bmm(H_post_norm.unsqueeze(2), layer_output.unsqueeze(1))
        
        # FIX 1: Convex Combination
        # alpha in (0, 1) ensures spectral norm of the operator is <= 1
        # (Assuming inputs are normalized, which they are via RMSNorm/LayerNorm)
        alpha = torch.sigmoid(self.stability_logit)
        
        x_next = alpha * mixed_stream + (1.0 - alpha) * write_back
        return x_next.view(B, T, N, D)


# ============================================================
# MoE Layer with mHC
# ============================================================

class mHCMoELayer(nn.Module):
    def __init__(self, hidden_dim, num_experts, n_expansion=4, k=1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.mhc = mHCConnection(hidden_dim, n_expansion)
        self.router = nn.Linear(hidden_dim, num_experts)
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.router.bias)

        self.register_buffer("router_mem", torch.zeros(num_experts))
        self.router_mem_decay = 0.9

        self.depth_gate = DepthGate(hidden_dim)

    def forward(self, x_stream, experts_pool, compute_entropy=True, update_router_mem=True):
        B, T, N, D = x_stream.shape
        
        H_pre, H_post, H_res, ent_pre, ent_post = self.mhc.compute_mappings(x_stream, compute_entropy)
        x_single = self.mhc.apply_pre(x_stream, H_pre)
        
        # Router
        # FIX 3: Removed additive bias `router_logits + self.router_mem`.
        # router_mem is updated for tracking/auxiliary loss but does not distort forward pass.
        router_logits = self.router(x_single)
        
        router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)
        probs = F.softmax(router_logits / 1.0, dim=-1)
        topk = probs.topk(self.k, dim=-1)
        
        output = torch.zeros_like(x_single)
        load = torch.zeros(self.num_experts, device=x_stream.device)
        
        for i in range(self.k):
            ids = topk.indices[:, i]
            weights = topk.values[:, i]
            for e_idx in range(self.num_experts):
                mask = ids == e_idx
                if not mask.any(): continue
                expert_out = experts_pool[e_idx](x_single[mask])
                if torch.isnan(expert_out).any():
                    expert_out = torch.nan_to_num(expert_out, nan=0.0)
                output[mask] += expert_out * weights[mask].unsqueeze(-1)
                load[e_idx] += (mask.float() * weights).sum()
        
        load = load / (load.sum() + 1e-8)
        
        if update_router_mem and self.training:
            with torch.no_grad():
                self.router_mem.mul_(self.router_mem_decay)
                self.router_mem.add_((1 - self.router_mem_decay) * load)

        load_balance_loss = (load * load).sum() * self.num_experts

        # FIX 4: Removed double application of DepthGate.
        # Gate is passed to mHC.apply_post_and_res, where it scales H_post (the connection strength).
        # We do NOT scale `output` directly here.
        gate = self.depth_gate(x_single)
        
        x_next = self.mhc.apply_post_and_res(x_stream, output, H_post, H_res, gate)
        
        return x_next, load_balance_loss, ent_pre, ent_post


# ============================================================
# Multi-Head Attention with RoPE and KV Cache
# ============================================================

class RoPEAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
    
    def forward(self, x, causal_mask, kv_cache=None, use_cache=False):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = self.rotary_emb(q, T)
        k = self.rotary_emb(k, T)
        
        if use_cache:
            if kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if causal_mask is not None:
            mask_size = scores.size(-1)
            if causal_mask.size(-1) < mask_size:
                extended_mask = torch.ones(
                    1, 1, scores.size(-2), mask_size,
                    device=causal_mask.device, dtype=torch.bool
                )
                extended_mask[:, :, :, -T:] = causal_mask[:, :, :T, :T]
                causal_mask = extended_mask
            scores = scores.masked_fill(causal_mask[:, :, :scores.size(-2), :scores.size(-1)], float('-inf'))
        
        scores = torch.clamp(scores, min=-1e4, max=1e4)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.o_proj(attn_output)
        
        if use_cache:
            return output, new_cache
        return output


# ============================================================
# Transformer Block with STABLE mHC
# ============================================================

class mHCTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_experts, n_expansion=4):
        super().__init__()
        self.n = n_expansion
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RoPEAttention(hidden_dim, num_heads)
        self.mhc_attn = mHCConnection(hidden_dim, n_expansion)
        
        self.depth_gate_attn = DepthGate(hidden_dim)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.moe = mHCMoELayer(hidden_dim, num_experts, n_expansion)

    def forward(self, x_stream, experts_pool, kv_cache=None, use_cache=False, compute_entropy=True, update_router_mem=True):
        B, T, N, D = x_stream.shape
        causal_mask = torch.triu(
            torch.ones(T, T, device=x_stream.device, dtype=torch.bool), 
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # --- Attention Path ---
        H_pre_attn, H_post_attn, H_res_attn, ent_pre_attn, ent_post_attn = \
            self.mhc_attn.compute_mappings(x_stream, compute_entropy)
        x_attn_in = self.mhc_attn.apply_pre(x_stream, H_pre_attn).view(B, T, D)
        
        if use_cache:
            attn_out, new_kv_cache = self.attn(
                self.norm1(x_attn_in), causal_mask, kv_cache, use_cache=True
            )
            attn_out = attn_out.view(B * T, D)
        else:
            attn_out = self.attn(
                self.norm1(x_attn_in), causal_mask, None, use_cache=False
            ).view(B * T, D)
            new_kv_cache = None
        
        # FIX 4: Consistent Depth Gate usage.
        # Compute gate, pass to mHC. Do not pre-scale attn_out.
        gate_attn = self.depth_gate_attn(x_attn_in.view(B * T, D))
        
        x_stream = self.mhc_attn.apply_post_and_res(x_stream, attn_out, H_post_attn, H_res_attn, gate_attn)
        
        # --- MoE Path ---
        x_moe_in = self.norm2(x_stream.view(B * T * N, D)).view(B, T, N, D)
        x_stream, lb_moe, ent_pre_moe, ent_post_moe = self.moe(x_moe_in, experts_pool, compute_entropy, update_router_mem)
        
        entropy_attn = ent_pre_attn + ent_post_attn
        entropy_moe = ent_pre_moe + ent_post_moe
        
        if use_cache:
            return x_stream, lb_moe, entropy_attn, entropy_moe, new_kv_cache
        return x_stream, lb_moe, entropy_attn, entropy_moe


# ============================================================
# Full Model
# ============================================================

class mHCExpertTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        num_experts=8,
        n_expansion=4,
        max_seq_len=2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_expansion = n_expansion
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        
        self.stream_expand = nn.Linear(hidden_dim, n_expansion * hidden_dim)
        nn.init.normal_(self.stream_expand.weight, std=0.02)
        nn.init.zeros_(self.stream_expand.bias)
        
        self.dropout = nn.Dropout(0.1)

        self.shared_experts = nn.ModuleList([
            SwiGLUExpert(hidden_dim, expansion_factor=4) for _ in range(num_experts)
        ])
        
        self.layers = nn.ModuleList([
            mHCTransformerBlock(hidden_dim, num_heads, num_experts, n_expansion)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.stream_compress = nn.Linear(n_expansion * hidden_dim, hidden_dim)
        nn.init.normal_(self.stream_compress.weight, std=0.02)
        nn.init.zeros_(self.stream_compress.bias)
        
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, input_ids, kv_caches=None, use_cache=False, compute_entropy=True, update_router_mem=True):
        B, T = input_ids.shape
        
        if (input_ids < 0).any() or (input_ids >= self.vocab_size).any():
            input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        x = self.embed(input_ids)
        x = self.dropout(x)
        
        x_expanded = self.stream_expand(x)
        x_stream = x_expanded.view(B, T, self.n_expansion, self.hidden_dim)
        
        total_lb = 0.0
        total_entropy_attn = 0.0
        total_entropy_moe = 0.0
        
        new_kv_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            
            if use_cache:
                x_stream, lb, e_attn, e_moe, new_kv_cache = layer(
                    x_stream, self.shared_experts, kv_cache, use_cache=True, 
                    compute_entropy=compute_entropy, update_router_mem=update_router_mem
                )
                new_kv_caches.append(new_kv_cache)
            else:
                x_stream, lb, e_attn, e_moe = layer(
                    x_stream, self.shared_experts, None, use_cache=False, 
                    compute_entropy=compute_entropy, update_router_mem=update_router_mem
                )
            
            total_lb += lb
            total_entropy_attn += e_attn
            total_entropy_moe += e_moe
            
            if torch.isnan(x_stream).any():
                x_stream = torch.nan_to_num(x_stream, nan=0.0)

        x_compressed = x_stream.view(B, T, -1)
        x_final = self.stream_compress(x_compressed)
        logits = self.lm_head(self.norm(x_final))
        logits = torch.clamp(logits, min=-1e4, max=1e4)
        
        if use_cache:
            return logits, total_lb, total_entropy_attn, total_entropy_moe, new_kv_caches
        return logits, total_lb, total_entropy_attn, total_entropy_moe


# ============================================================
# Script entry point
# ============================================================

def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    print("="*60)
    print(" Creating STABLE mHC Expert Transformer")
    print(" Fixes Applied:")
    print("  ✅ Spectral Norm <= 1 (Convex Combination)")
    print("  ✅ Sinkhorn FP32 Stability")
    print("  ✅ Router Memory Cleaned (No Bias)")
    print("  ✅ Depth Gate Consistent")
    print("="*60)
    print(f" Vocab size: {vocab_size:,}")
    print(f" Hidden dim: 512")
    print(f" Layers: 8")
    print(f" Heads: 8")
    print(f" Experts: 8 (Shared SwiGLU)")
    print(f" Stream expansion: 4")
    print(f" Sinkhorn iterations: 20")
    print(f" Precision: bfloat16")
    print("="*60)

    model = mHCExpertTransformer(
        vocab_size=vocab_size,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        num_experts=8,
        n_expansion=4,
        max_seq_len=2048,
    )
    
    model = model.to(dtype=torch.bfloat16)

    save_path = Path("mhc_stable_transformer.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "hidden_dim": 512,
                "num_layers": 8,
                "num_heads": 8,
                "num_experts": 8,
                "n_expansion": 4,
                "max_seq_len": 2048,
                "architecture": "mhc_stable_v2"
            },
            "tokenizer_name": "gpt2",
            "dtype": "bfloat16",
            "version": "stable_mhc_v2_fixed"
        },
        save_path,
    )

    print(f"\n[SUCCESS] STABLE mHC Model saved to: {save_path.resolve()}")
    
    total_params = sum(p.numel() for p in model.parameters())
    import os
    actual_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Saved as: bfloat16")
    print(f"  Actual file size: {actual_size_mb:.1f} MB")
    print(f"  Architecture: STABLE mHC (Convex Combo + FP32 Sinkhorn)")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR in script execution: {e}")
        import traceback
        traceback.print_exc()