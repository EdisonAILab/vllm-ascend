"""Debug silu_and_mul with H=4096 (same as test)."""
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ops.triton.batch_invariant.silu_and_mul import silu_and_mul_batch_invariant

DEVICE = "npu"
DTYPE = torch.bfloat16

gen = torch.Generator().manual_seed(42)
H = 4096
x = torch.randn(4, 2 * H, generator=gen, dtype=DTYPE).to(DEVICE)

out_triton = silu_and_mul_batch_invariant(x)
out_ref = F.silu(x[:, :H]) * x[:, H:]

diff = (out_triton.float() - out_ref.float()).abs()
max_diff = diff.max().item()
print("Shape:", out_triton.shape, "H:", H)
print("Max diff:", max_diff)

# Find where the largest diffs are
flat = diff.flatten()
top10 = torch.topk(flat, 10)
print("Top 10 diffs:", top10.values.tolist())
print("At indices:", top10.indices.tolist())

# Check row 0 in detail
for col_start in [0, 1024, 2048, 3072]:
    chunk_diff = diff[0, col_start:col_start+10]
    print("Row 0, cols {}-{}: {}".format(col_start, col_start+10, chunk_diff.tolist()))

# Test with H=128 (small)
x_small = torch.randn(4, 256, generator=gen, dtype=DTYPE).to(DEVICE)
out_s = silu_and_mul_batch_invariant(x_small)
ref_s = F.silu(x_small[:, :128]) * x_small[:, 128:]
print("\nH=128 max_diff:", (out_s.float() - ref_s.float()).abs().max().item())

# Test with H=1024
x_mid = torch.randn(4, 2048, generator=gen, dtype=DTYPE).to(DEVICE)
out_m = silu_and_mul_batch_invariant(x_mid)
ref_m = F.silu(x_mid[:, :1024]) * x_mid[:, 1024:]
print("H=1024 max_diff:", (out_m.float() - ref_m.float()).abs().max().item())

# BI test: same input, different chunks
x2 = torch.randn(8, 2 * H, generator=gen, dtype=DTYPE).to(DEVICE)
full = silu_and_mul_batch_invariant(x2)
h1 = silu_and_mul_batch_invariant(x2[:4])
h2 = silu_and_mul_batch_invariant(x2[4:])
cat = torch.cat([h1, h2])
print("\nBI (split 4+4):", torch.equal(full, cat))

h1 = silu_and_mul_batch_invariant(x2[:1])
rest = silu_and_mul_batch_invariant(x2[1:])
cat2 = torch.cat([h1, rest])
print("BI (split 1+7):", torch.equal(full, cat2))
