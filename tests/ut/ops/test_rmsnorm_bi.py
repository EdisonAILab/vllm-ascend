"""Test npu_rms_norm and npu_add_rms_norm batch-invariance on Ascend NPU."""
import torch
import torch_npu

DEVICE = "npu"
DTYPE = torch.bfloat16

print("=" * 60)
print("Test: npu_rms_norm / npu_add_rms_norm batch-invariance")
print("=" * 60)

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        print("  [PASS] {}".format(name))
        passed += 1
    else:
        print("  [FAIL] {}".format(name))
        failed += 1


# ── Test 1: npu_rms_norm with different batch sizes ──
print("\n--- Test 1: npu_rms_norm with different batch sizes ---")

hidden = 4096
eps = 1e-6
gen = torch.Generator().manual_seed(42)
weight = torch.randn(hidden, generator=gen, dtype=DTYPE).to(DEVICE)
x_large = torch.randn(256, hidden, generator=gen, dtype=DTYPE).to(DEVICE)

out_full, _ = torch_npu.npu_rms_norm(x_large, weight, eps)

for bs in [1, 3, 7, 16, 32, 64, 128]:
    chunks = []
    for s in range(0, 256, bs):
        e = min(s + bs, 256)
        o, _ = torch_npu.npu_rms_norm(x_large[s:e], weight, eps)
        chunks.append(o)
    out_chunked = torch.cat(chunks, dim=0)
    match = torch.equal(out_full, out_chunked)
    if not match:
        dc = (out_full != out_chunked).sum().item()
        tt = out_full.numel()
        md = (out_full - out_chunked).abs().max().item()
        check("BS={} (diff={}/{}, max={:.4e})".format(bs, dc, tt, md), False)
    else:
        check("BS={}".format(bs), True)

# ── Test 2: npu_rms_norm with model-scale dimensions ──
print("\n--- Test 2: Model-scale dimensions ---")

for hidden_dim in [2048, 4096, 7168, 14336]:
    gen2 = torch.Generator().manual_seed(42)
    w = torch.randn(hidden_dim, generator=gen2, dtype=DTYPE).to(DEVICE)
    x = torch.randn(128, hidden_dim, generator=gen2, dtype=DTYPE).to(DEVICE)

    out_f, _ = torch_npu.npu_rms_norm(x, w, eps)

    rows = []
    for i in range(128):
        o, _ = torch_npu.npu_rms_norm(x[i:i + 1], w, eps)
        rows.append(o)
    out_single = torch.cat(rows, dim=0)

    match = torch.equal(out_f, out_single)
    if not match:
        dc = (out_f != out_single).sum().item()
        tt = out_f.numel()
        md = (out_f - out_single).abs().max().item()
        check("hidden={} (diff={}/{}, max={:.4e})".format(hidden_dim, dc, tt, md), False)
    else:
        check("hidden={}".format(hidden_dim), True)

# ── Test 3: npu_add_rms_norm (fused) batch-invariance ──
print("\n--- Test 3: npu_add_rms_norm (fused) batch-invariance ---")

hidden = 4096
gen3 = torch.Generator().manual_seed(42)
w3 = torch.randn(hidden, generator=gen3, dtype=DTYPE).to(DEVICE)
x3 = torch.randn(128, hidden, generator=gen3, dtype=DTYPE).to(DEVICE)
residual3 = torch.randn(128, hidden, generator=gen3, dtype=DTYPE).to(DEVICE)

out_full3, _, res_full3 = torch_npu.npu_add_rms_norm(x3, residual3, w3, eps)

for bs in [1, 7, 16, 32, 64]:
    out_chunks = []
    res_chunks = []
    for s in range(0, 128, bs):
        e = min(s + bs, 128)
        o, _, r = torch_npu.npu_add_rms_norm(x3[s:e], residual3[s:e], w3, eps)
        out_chunks.append(o)
        res_chunks.append(r)
    out_c3 = torch.cat(out_chunks, dim=0)
    res_c3 = torch.cat(res_chunks, dim=0)

    out_match = torch.equal(out_full3, out_c3)
    res_match = torch.equal(res_full3, res_c3)

    if out_match and res_match:
        check("npu_add_rms_norm BS={}".format(bs), True)
    else:
        parts = []
        if not out_match:
            dc = (out_full3 != out_c3).sum().item()
            tt = out_full3.numel()
            md = (out_full3 - out_c3).abs().max().item()
            parts.append("out_diff={}/{} max={:.4e}".format(dc, tt, md))
        if not res_match:
            dc = (res_full3 != res_c3).sum().item()
            parts.append("res_diff={}".format(dc))
        check("npu_add_rms_norm BS={} ({})".format(bs, ", ".join(parts)), False)

# ── Test 4: Split approach (add + rms_norm) ──
print("\n--- Test 4: Split add + npu_rms_norm batch-invariance ---")


def add_rms_norm_split(x, residual, w, epsilon):
    x_ = x + residual
    residual_ = x_
    x_, _ = torch_npu.npu_rms_norm(x_, w, epsilon)
    return x_, None, residual_


out_sf, _, res_sf = add_rms_norm_split(x3, residual3, w3, eps)

for bs in [1, 7, 16, 32, 64]:
    out_chunks = []
    res_chunks = []
    for s in range(0, 128, bs):
        e = min(s + bs, 128)
        o, _, r = add_rms_norm_split(x3[s:e], residual3[s:e], w3, eps)
        out_chunks.append(o)
        res_chunks.append(r)
    out_sc = torch.cat(out_chunks, dim=0)
    res_sc = torch.cat(res_chunks, dim=0)

    out_match = torch.equal(out_sf, out_sc)
    res_match = torch.equal(res_sf, res_sc)

    if out_match and res_match:
        check("split add+rms_norm BS={}".format(bs), True)
    else:
        parts = []
        if not out_match:
            dc = (out_sf != out_sc).sum().item()
            tt = out_sf.numel()
            md = (out_sf - out_sc).abs().max().item()
            parts.append("out_diff={}/{} max={:.4e}".format(dc, tt, md))
        if not res_match:
            dc = (res_sf != res_sc).sum().item()
            parts.append("res_diff={}".format(dc))
        check("split add+rms_norm BS={} ({})".format(bs, ", ".join(parts)), False)

# ── Test 5: Determinism ──
print("\n--- Test 5: Determinism (5 runs) ---")
ref, _ = torch_npu.npu_rms_norm(x_large, weight, eps)
all_det = True
for i in range(5):
    o, _ = torch_npu.npu_rms_norm(x_large, weight, eps)
    if not torch.equal(o, ref):
        all_det = False
        break
check("npu_rms_norm deterministic (5 runs)", all_det)

# Summary
print("\n" + "=" * 60)
total = passed + failed
status = "ALL PASSED" if failed == 0 else "{} FAILED".format(failed)
print("  Results: {}/{} passed - {}".format(passed, total, status))
print("=" * 60)
