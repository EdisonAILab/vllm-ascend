"""Debug silu_and_mul Triton kernel."""
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ops.triton.batch_invariant.silu_and_mul import silu_and_mul_batch_invariant

DEVICE = "npu"
DTYPE = torch.bfloat16

# Small tensor for debugging
torch.manual_seed(42)
x = torch.randn(2, 8, dtype=DTYPE).to(DEVICE)
H = 4

print("Input x:")
print(x)
print()

out_triton = silu_and_mul_batch_invariant(x)
print("Triton output:")
print(out_triton)

gate = x[:, :H]
up = x[:, H:]
out_ref = F.silu(gate) * up
print("\nReference output:")
print(out_ref)

print("\nDiff:")
print((out_triton.float() - out_ref.float()).abs())
print("Max diff:", (out_triton.float() - out_ref.float()).abs().max().item())

# Check if gate/up split is correct
print("\nGate (x[:,:H]):", gate)
print("Up (x[:,H:]):", up)
print("SiLU(gate):", F.silu(gate))
