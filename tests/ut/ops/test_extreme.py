"""Extreme dimension test for npu_quant_matmul batch-invariance.

Tests whether batch-invariance holds under:
1. Very large K (split-K tiling pressure)
2. Very large M (batch scheduling pressure)
3. Non-aligned dimensions (boundary handling)
4. Unusual aspect ratios
5. DeepSeek V3 realistic dimensions
"""
import torch
import torch_npu

DEVICE = "npu"
DTYPE = torch.bfloat16
E8M0 = torch_npu.float8_e8m0fnu
GS = 32

passed = 0
failed = 0


def check(name, condition, extra=""):
    global passed, failed
    if condition:
        print("  [PASS] {}".format(name))
        passed += 1
    else:
        print("  [FAIL] {} {}".format(name, extra))
        failed += 1


def make_weight(K, N, seed=123):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=gen, dtype=DTYPE).to(DEVICE)
    wf, ws = torch_npu.npu_dynamic_mx_quant(w, dst_type=torch.float8_e4m3fn)
    wf_t = wf.view(torch.uint8).transpose(0, 1).contiguous().view(torch.float8_e4m3fn)
    ws_t = ws.transpose(0, 1).contiguous()
    return wf_t, ws_t


def test_bi(name, M_total, K, N, batch_sizes=None):
    """Test batch-invariance for given dimensions."""
    if batch_sizes is None:
        batch_sizes = [1, 7, 16, 64]

    gen = torch.Generator().manual_seed(42)
    x = torch.randn(M_total, K, generator=gen, dtype=DTYPE).to(DEVICE)
    wf, ws = make_weight(K, N)

    qx, sx = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
    out_full = torch_npu.npu_quant_matmul(
        qx, wf, ws, scale_dtype=E8M0, pertoken_scale=sx,
        pertoken_scale_dtype=E8M0, output_dtype=DTYPE, group_sizes=[1, 1, GS],
    )

    for bs in batch_sizes:
        chunks = []
        for s in range(0, M_total, bs):
            e = min(s + bs, M_total)
            qi = qx.view(torch.uint8)[s:e].view(torch.float8_e4m3fn)
            si = sx[s:e]
            oi = torch_npu.npu_quant_matmul(
                qi, wf, ws, scale_dtype=E8M0, pertoken_scale=si,
                pertoken_scale_dtype=E8M0, output_dtype=DTYPE, group_sizes=[1, 1, GS],
            )
            chunks.append(oi)
        out_bs = torch.cat(chunks, dim=0)
        match = torch.equal(out_full, out_bs)
        if not match:
            dc = (out_full != out_bs).sum().item()
            tt = out_full.numel()
            md = (out_full - out_bs).abs().max().item()
            check("{} BS={}".format(name, bs), False,
                  "diff={}/{} ({:.2f}%) max={:.4e}".format(dc, tt, dc / tt * 100, md))
        else:
            check("{} BS={}".format(name, bs), True)


print("=" * 60)
print("Extreme dimension batch-invariance test")
print("=" * 60)

# ── 1. Very large K (split-K pressure) ──
print("\n--- 1. Very large K ---")
test_bi("K=16384", M_total=64, K=16384, N=2048)
test_bi("K=32768", M_total=64, K=32768, N=2048)
try:
    test_bi("K=65536", M_total=32, K=65536, N=1024)
except Exception as e:
    print("  [SKIP] K=65536: {}".format(str(e)[:80]))

# ── 2. Very large M (batch scheduling pressure) ──
print("\n--- 2. Very large M ---")
test_bi("M=1024", M_total=1024, K=4096, N=4096, batch_sizes=[1, 32, 128, 512])
test_bi("M=4096", M_total=4096, K=4096, N=4096, batch_sizes=[1, 64, 256, 1024])
try:
    test_bi("M=8192", M_total=8192, K=4096, N=2048, batch_sizes=[1, 128, 512, 2048])
except Exception as e:
    print("  [SKIP] M=8192: {}".format(str(e)[:80]))

# ── 3. Non-aligned dimensions ──
print("\n--- 3. Non-aligned dimensions ---")
test_bi("K=4128(129*32)", M_total=64, K=4128, N=2048)
test_bi("K=4064(127*32)", M_total=64, K=4064, N=2048)
test_bi("N=2049(odd)", M_total=64, K=4096, N=2049)
test_bi("N=1023(odd)", M_total=64, K=4096, N=1023)
test_bi("M=63(odd)", M_total=63, K=4096, N=2048, batch_sizes=[1, 7, 16])
test_bi("M=1(single)", M_total=1, K=4096, N=2048, batch_sizes=[1])

# ── 4. Unusual aspect ratios ──
print("\n--- 4. Unusual aspect ratios ---")
test_bi("tall: M=4096,K=128,N=128", M_total=4096, K=128, N=128, batch_sizes=[1, 32, 256])
test_bi("wide: M=4,K=4096,N=32768", M_total=4, K=4096, N=32768, batch_sizes=[1, 2])
try:
    test_bi("skinny: M=256,K=32,N=4096", M_total=256, K=32, N=4096, batch_sizes=[1, 16, 64])
except Exception as e:
    print("  [SKIP] skinny K=32: {}".format(str(e).split("\\n")[0][:80]))
test_bi("square: M=256,K=4096,N=4096", M_total=256, K=4096, N=4096, batch_sizes=[1, 32, 128])

# ── 5. DeepSeek V3 / MoE realistic dims ──
print("\n--- 5. DeepSeek V3 realistic dimensions ---")
test_bi("DSv3 gate_up: K=7168,N=18432", M_total=256, K=7168, N=18432,
        batch_sizes=[1, 7, 32, 128])
test_bi("DSv3 down: K=9216,N=7168", M_total=256, K=9216, N=7168,
        batch_sizes=[1, 7, 32, 128])
test_bi("DSv3 qkv: K=7168,N=1536", M_total=512, K=7168, N=1536,
        batch_sizes=[1, 16, 64, 256])

# Summary
print("\n" + "=" * 60)
total = passed + failed
status = "ALL PASSED" if failed == 0 else "{} FAILED".format(failed)
print("  Results: {}/{} passed - {}".format(passed, total, status))
print("=" * 60)
