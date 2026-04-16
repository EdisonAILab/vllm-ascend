"""Verify the new Triton BI kernels: BI property + correctness vs reference + benchmark."""
import time
import torch
import torch.nn.functional as F
import torch_npu

# Add vllm_ascend to path if needed
import sys
sys.path.insert(0, "/home/o00649568/b84411271")

DEVICE = "npu"
DTYPE = torch.bfloat16

WARMUP = 5
ITERS = 30


def benchmark(fn, *args, **kwargs):
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = fn(*args, **kwargs)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000, out


passed, failed = 0, 0


def check(name, condition, extra=""):
    global passed, failed
    if condition:
        print("  [PASS] {}".format(name))
        passed += 1
    else:
        print("  [FAIL] {} {}".format(name, extra))
        failed += 1


def equal_or_tuple(a, b):
    if isinstance(a, tuple):
        return all(torch.equal(x, y) for x, y in zip(a, b))
    return torch.equal(a, b)


def cat_or_tuple(chunks):
    if isinstance(chunks[0], tuple):
        return tuple(torch.cat([c[i] for c in chunks], dim=0)
                     for i in range(len(chunks[0])))
    return torch.cat(chunks, dim=0)


def test_bi(name, fn, args_factory, M_total, batch_sizes):
    """Test BI: split args[0] (the M dim) into chunks."""
    args_full = args_factory(M_total)
    out_full = fn(*args_full)

    all_ok = True
    for bs in batch_sizes:
        chunks = []
        for s in range(0, M_total, bs):
            e = min(s + bs, M_total)
            args_chunk = tuple(
                a[s:e].contiguous() if isinstance(a, torch.Tensor) and a.shape[0] == M_total else a
                for a in args_full
            )
            chunks.append(fn(*args_chunk))
        out_cat = cat_or_tuple(chunks)
        if not equal_or_tuple(out_full, out_cat):
            all_ok = False
            print("    BS={}: MISMATCH".format(bs))
    check("{} batch-invariant".format(name), all_ok)


print("=" * 60)
print("Triton BI Kernels: BI verification + correctness + benchmark")
print("=" * 60)

# ============================================================
# 1. silu_and_mul
# ============================================================
print("\n--- silu_and_mul ---")
from vllm_ascend.ops.triton.batch_invariant.silu_and_mul import silu_and_mul_batch_invariant


def silu_native(x):
    H = x.shape[-1] // 2
    return F.silu(x[..., :H]) * x[..., H:]


def make_silu(M):
    H = 4096
    gen = torch.Generator().manual_seed(42)
    return (torch.randn(M, 2 * H, generator=gen, dtype=DTYPE).to(DEVICE),)


test_bi("silu_and_mul", silu_and_mul_batch_invariant, make_silu,
        M_total=256, batch_sizes=[1, 7, 32, 64, 128])
_silu_args = make_silu(256)
out_bi = silu_and_mul_batch_invariant(*_silu_args)
out_ref = silu_native(*_silu_args)
max_d = (out_bi.float() - out_ref.float()).abs().max().item()
check("silu_and_mul correctness vs PyTorch (max_diff={:.2e})".format(max_d), max_d < 5e-2)
for M in [16, 256, 4096]:
    args = make_silu(M)
    t_ref, _ = benchmark(silu_native, *args)
    t_bi, _ = benchmark(silu_and_mul_batch_invariant, *args)
    print("  M={:>4d}: ref={:.3f}ms  triton={:.3f}ms  speedup={:.2f}x".format(
        M, t_ref, t_bi, t_ref / t_bi))


# ============================================================
# 2. logsoftmax
# ============================================================
print("\n--- logsoftmax ---")
from vllm_ascend.ops.triton.batch_invariant.logsoftmax import logsoftmax_batch_invariant


def make_logsoftmax(M):
    vocab = 32000
    gen = torch.Generator().manual_seed(42)
    return (torch.randn(M, vocab, generator=gen, dtype=DTYPE).to(DEVICE),)


test_bi("logsoftmax", logsoftmax_batch_invariant, make_logsoftmax,
        M_total=256, batch_sizes=[1, 7, 32, 64, 128])
args = make_logsoftmax(256)
out_bi = logsoftmax_batch_invariant(*args)
out_ref = F.log_softmax(args[0], dim=-1)
max_d = (out_bi.float() - out_ref.float()).abs().max().item()
check("logsoftmax correctness vs PyTorch (max_diff={:.2e})".format(max_d), max_d < 1e-1)
for M in [16, 256, 4096]:
    args = make_logsoftmax(M)
    t_ref, _ = benchmark(F.log_softmax, args[0], dim=-1)
    t_bi, _ = benchmark(logsoftmax_batch_invariant, *args)
    print("  M={:>4d}: ref={:.3f}ms  triton={:.3f}ms  speedup={:.2f}x".format(
        M, t_ref, t_bi, t_ref / t_bi))


# ============================================================
# 3. topk_softmax
# ============================================================
print("\n--- topk_softmax ---")
from vllm_ascend.ops.triton.batch_invariant.topk_softmax import topk_softmax_batch_invariant


def make_topk(M):
    n_experts = 8
    gen = torch.Generator().manual_seed(42)
    return (torch.randn(M, n_experts, generator=gen, dtype=DTYPE).to(DEVICE), 2)


def topk_softmax_ref(scores, k):
    probs = F.softmax(scores, dim=-1)
    return torch.topk(probs, k=k, dim=-1)


test_bi("topk_softmax", topk_softmax_batch_invariant, make_topk,
        M_total=256, batch_sizes=[1, 7, 32, 64, 128])
args = make_topk(256)
weights_bi, indices_bi = topk_softmax_batch_invariant(*args)
ref_weights, ref_indices = topk_softmax_ref(*args)
max_dw = (weights_bi.float() - ref_weights.float()).abs().max().item()
ids_match = torch.equal(indices_bi.to(torch.int64), ref_indices)
check("topk_softmax correctness (weight max_diff={:.2e}, idx_match={})".format(
    max_dw, ids_match), max_dw < 1e-2 and ids_match)
for M in [16, 256, 4096]:
    args = make_topk(M)
    t_ref, _ = benchmark(topk_softmax_ref, *args)
    t_bi, _ = benchmark(topk_softmax_batch_invariant, *args)
    print("  M={:>4d}: ref={:.3f}ms  triton={:.3f}ms  speedup={:.2f}x".format(
        M, t_ref, t_bi, t_ref / t_bi))


# ============================================================
# 4. moe_gating
# ============================================================
print("\n--- moe_gating ---")
from vllm_ascend.ops.triton.batch_invariant.moe_gating import moe_gating_batch_invariant


def make_moe(M):
    hidden = 4096
    n_experts = 8
    gen = torch.Generator().manual_seed(42)
    h = torch.randn(M, hidden, generator=gen, dtype=DTYPE).to(DEVICE)
    w = torch.randn(n_experts, hidden, generator=gen, dtype=DTYPE).to(DEVICE)
    return (h, w, 2)


def moe_gating_ref(h, w, k):
    logits = h @ w.T
    probs = F.softmax(logits, dim=-1)
    return torch.topk(probs, k=k, dim=-1)


test_bi("moe_gating", moe_gating_batch_invariant, make_moe,
        M_total=256, batch_sizes=[1, 7, 32, 64, 128])
args = make_moe(256)
weights_bi, indices_bi = moe_gating_batch_invariant(*args)
ref_w, ref_i = moe_gating_ref(*args)
ids_match = torch.equal(indices_bi.to(torch.int64), ref_i)
check("moe_gating correctness (idx_match={})".format(ids_match), ids_match)
for M in [16, 256, 4096]:
    args = make_moe(M)
    t_ref, _ = benchmark(moe_gating_ref, *args)
    t_bi, _ = benchmark(moe_gating_batch_invariant, *args)
    print("  M={:>4d}: ref={:.3f}ms  triton={:.3f}ms  speedup={:.2f}x".format(
        M, t_ref, t_bi, t_ref / t_bi))


# ============================================================
# 5. rotary_embedding
# ============================================================
print("\n--- rotary_embedding ---")
from vllm_ascend.ops.triton.batch_invariant.rotary_embedding import rotary_embedding_batch_invariant


def make_rotary(M):
    head_dim = 128
    n_q_heads = 32
    n_k_heads = 32
    gen = torch.Generator().manual_seed(42)
    q = torch.randn(M, n_q_heads * head_dim, generator=gen, dtype=DTYPE).to(DEVICE)
    k = torch.randn(M, n_k_heads * head_dim, generator=gen, dtype=DTYPE).to(DEVICE)
    cos = torch.randn(M, head_dim, generator=gen, dtype=DTYPE).to(DEVICE)
    sin = torch.randn(M, head_dim, generator=gen, dtype=DTYPE).to(DEVICE)
    return (q, k, cos, sin, head_dim)


def rotary_ref(q, k, cos, sin, head_dim):
    """PyTorch reference (NEOX style)."""
    def rope(x):
        # x: [M, n_heads * head_dim] -> reshape to [M, n_heads, head_dim]
        M = x.shape[0]
        x_r = x.view(M, -1, head_dim)
        a = x_r[..., :head_dim // 2]
        b = x_r[..., head_dim // 2:]
        cos_a = cos[:, :head_dim // 2].unsqueeze(1)
        cos_b = cos[:, head_dim // 2:].unsqueeze(1)
        sin_a = sin[:, :head_dim // 2].unsqueeze(1)
        sin_b = sin[:, head_dim // 2:].unsqueeze(1)
        out_a = a * cos_a - b * sin_a
        out_b = b * cos_b + a * sin_b
        out = torch.cat([out_a, out_b], dim=-1)
        return out.reshape(M, -1)
    return rope(q.clone()), rope(k.clone())


# Note: Triton kernel modifies q/k in-place, so we need fresh copies for BI test
def rotary_bi_wrapper(q, k, cos, sin, head_dim):
    q_c = q.clone()
    k_c = k.clone()
    return rotary_embedding_batch_invariant(q_c, k_c, cos, sin, head_dim)


test_bi("rotary_embedding", rotary_bi_wrapper, make_rotary,
        M_total=256, batch_sizes=[1, 7, 32, 64, 128])

_rotary_args = make_rotary(256)
q_bi, k_bi = rotary_bi_wrapper(*_rotary_args)
q_ref, k_ref = rotary_ref(*_rotary_args)
max_dq = (q_bi.float() - q_ref.float()).abs().max().item()
max_dk = (k_bi.float() - k_ref.float()).abs().max().item()
check("rotary correctness (q={:.2e}, k={:.2e})".format(max_dq, max_dk),
      max_dq < 5e-2 and max_dk < 5e-2)

for M in [16, 256, 4096]:
    args = make_rotary(M)
    t_ref, _ = benchmark(rotary_ref, *args)
    t_bi, _ = benchmark(rotary_bi_wrapper, *args)
    print("  M={:>4d}: ref={:.3f}ms  triton={:.3f}ms  speedup={:.2f}x".format(
        M, t_ref, t_bi, t_ref / t_bi))


# Summary
print("\n" + "=" * 60)
total = passed + failed
status = "ALL PASSED" if failed == 0 else "{} FAILED".format(failed)
print("  Results: {}/{} - {}".format(passed, total, status))
print("=" * 60)
