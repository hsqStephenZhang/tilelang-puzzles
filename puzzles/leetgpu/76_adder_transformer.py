r"""
LeetGPU 76: Adder Transformer Inference
=======================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Run batched autoregressive inference for a 10-parameter transformer that adds two 10-digit
numbers. Given prompts of shape `[batch_size, 31]` (int32) and a 10-float weight
buffer, write output logits of shape `[batch_size, 11, 10]` — one logit
row per decode step over the 10-digit vocabulary (0–9). All tensors are float32 except
the int32 prompts.

The model comes from the
AdderBoard
competition for the smallest autoregressive transformer that adds 10-digit numbers at
≥99% accuracy. It encodes carry propagation in 10 learned parameters via RoPE geometry,
tied embeddings, and SwiGLU gating.

(diagram omitted, see challenge.html in leetgpu-challenges)

Model Architecture
------------------

Single-layer pre-norm transformer. Hidden dim 2, 1 head, head dim 2, vocab 10 (digits
0–9), tied input/output embeddings.

Each step runs the full sequence `[batch_size, seq_len, 2]` through:

1. Token Embedding (2 parameters: `w0`, `w1`)

$$e(d) = \begin{bmatrix} w_0 - w_1 \cdot d^2 \\ -d \end{bmatrix}$$

2. Unit RMSNorm (no parameters)

$$\text{UnitRMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}, \quad \epsilon = 10^{-6}$$

3. Self-Attention (3 parameters: `q0`, `q1`, `v0`)

Projections applied to the normed hidden state `h` with shape `[*, 2]`:

$$Q = \begin{bmatrix} h_0 \cdot q_0 \\ h_0 \cdot q_1 \end{bmatrix}, \quad
K = \begin{bmatrix} h_0 \\ 0 \end{bmatrix}, \quad
V = \begin{bmatrix} h_1 \cdot v_0 \\ 0 \end{bmatrix}$$

After projection, Q and K are each normalized with Unit RMSNorm, then RoPE is applied
with angular frequency `ω = 2π/19`:

$$\text{RoPE}(x, p) = \begin{bmatrix} x_0 \cos(p\omega) - x_1 \sin(p\omega) \\
x_0 \sin(p\omega) + x_1 \cos(p\omega) \end{bmatrix}$$

Scaled dot-product attention with causal mask uses scale factor:

$$\text{scale} = \frac{1}{\sqrt{d_h}} \cdot S^2$$

where \(d_h = 2\) is the head dimension and \(S^2\) is the QK-norm scale constant
(see weight table below for exact value).

The output projection maps `[attn_0, attn_1]` → `[0, attn_0]`
(no parameters), followed by a residual connection.

4. MLP (3 parameters: `a`, `c`, `carry`)

Applied to the unit-RMSNorm of the post-attention hidden state:

$$g_0 = h_0 \cdot a + h_1 \cdot c, \quad g_1 = h_0 \cdot (a - c / 1000) + h_1 \cdot c$$

$$\text{base} = h_0, \quad \text{up} = [\text{base}, \text{base}]$$

$$\text{mix} = \text{SiLU}([g_0, g_1]) \odot \text{up}$$

$$\text{MLP}(h) = \begin{bmatrix} 0 \\ \text{carry} \cdot (\text{mix}_1 - \text{mix}_0) \end{bmatrix}$$

followed by a residual connection.

5. Final RMSNorm (2 parameters: `n0`, `n1`)

Standard RMSNorm with learned weight:

$$\text{out} = \frac{h}{\sqrt{\text{mean}(h^2) + \epsilon}} \odot [n_0, n_1]$$

6. Output Logits (tied with embedding)

$$\text{logits} = \text{out} \cdot E^T \quad \text{where } E_{d} = e(d)$$

Autoregressive Decoding
-----------------------

Starting from the 31-token prompt, repeat 11 times:

- Run the full forward pass on the current sequence
- Extract logits at the last position → store in output
- Append `argmax(logits)` as the next token

The sequence grows from length 31 to 42 over the 11 decode steps.

Weight Layout
-------------

| Offset | Size | Name | Description

| 0 | 2 | embed | Embedding: `e(d) = [w0 - w1*d², -d]`

| 2 | 2 | q_proj | Q projection weights `[q0, q1]`

| 4 | 1 | v_proj | V projection weight `v0`

| 5 | 2 | gate | MLP gate weights `[a, c]`

| 7 | 1 | carry | MLP carry weight

| 8 | 2 | norm | Final RMSNorm weight `[n0, n1]`

Token Encoding
--------------

Each input pair `(a, b)` of 10-digit numbers is encoded as a 31-token sequence:

[0, a_rev_0, ..., a_rev_9, 0, 0, 0, 0, 0, 0, 0, 0, 0, b_rev_0, ..., b_rev_9, 0]

where `a_rev` and `b_rev` are the digits in least-significant-first order,
zero-padded to 10 digits. The model then generates 11 output tokens (digits of the sum, also
least-significant-first).

Implementation Requirements
---------------------------

- Implement `solve(prompts, output, weights, batch_size)` with the exact signature shown (JAX exception: `solve(prompts, weights, batch_size)` returns the output tensor directly)
- Do not use any external libraries beyond what the framework provides
- The function must write logits into the `output` buffer (except JAX, which returns it)
- Architecture constants are fixed: `vocab_size` = 10, `hidden_dim` = 2,
`head_dim` = 2, `num_heads` = 1, `prompt_len` = 31,
`decode_steps` = 11

- RMSNorm epsilon = 10−6
- RoPE angular frequency ω = 2π/19
- Attention scale = (1/√2) · `S`² where `S`² = ln(10) / (√2 · (cos(0.3ω) − cos(0.7ω)))
- SiLU activation: `silu(x) = x · sigmoid(x)`

Example
-------

With `batch_size` = 2 and pairs (3, 5), (99, 1):

Input prompts (shape [2, 31]):
[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Output logits shape: [2, 11, 10]
(logits at each of 11 decode steps over 10 digit classes)

Expected decoded tokens (via argmax):
Pair (3, 5):   sum = 8       → [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Pair (99, 1):  sum = 100     → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

Constraints
-----------

- `batch_size`: 1 ≤ `batch_size` ≤ 100,000
- `prompts`: 32-bit integer tensor, values in [0, 9]
- `weights`: 32-bit float tensor with exactly 10 elements
- `output`: 32-bit float tensor of shape `[batch_size, 11, 10]`
- Input numbers are in range [0, 9,999,999,999] (10-digit unsigned integers)
- Performance is measured with `batch_size` = 100,000

Run `python3 puzzles/leetgpu/76_adder_transformer.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
import math
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
import torch.nn.functional as F
from common.leetgpu import ChallengeBase, run_challenge

# Model architecture constants
VOCAB_SIZE = 10
MODEL_DIM = 2
HEAD_DIM = 2
PROMPT_LEN = 31
OUTPUT_DIGITS = 11
RMS_EPS = 1e-6

# Derived constants from the hand-crafted 10-parameter adder model
EMBED_CONST = 1000.0
CONST_NORM = math.sqrt(MODEL_DIM)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
DECODE_QUAD = 1e-3
DECODE_CURVATURE = 0.1
ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)
TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_NORM_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))
CARRY_ALPHA = 256.0 / CONST_NORM
ATTN_SCALE = (HEAD_DIM**-0.5) * (QK_NORM_SCALE**2)

# Weight buffer layout (10 parameters total)
O_EMBED = 0  # [2] embedding: e(d) = [w0 - w1*d^2, -d]
O_QPROJ = 2  # [2] Q projection weights
O_VPROJ = 4  # [1] V projection weight
O_GATE = 5  # [2] MLP gate weights
O_CARRY = 7  # [1] MLP carry weight
O_NORM = 8  # [2] final RMSNorm weight
TOTAL_WEIGHTS = 10


def _encode_pair(a: int, b: int) -> list:
    a_digits = [int(c) for c in f"{a:010d}"][::-1]
    b_digits = [int(c) for c in f"{b:010d}"][::-1]
    return [0] + a_digits + [0] * 9 + b_digits + [0]


def _encode_pairs_batch(a_vals: torch.Tensor, b_vals: torch.Tensor, device) -> torch.Tensor:
    batch_size = a_vals.shape[0]
    prompts = torch.zeros(batch_size, PROMPT_LEN, device=device, dtype=torch.int32)
    a = a_vals.clone().to(torch.int64)
    for i in range(10):
        prompts[:, 1 + i] = (a % 10).to(torch.int32)
        a = a // 10
    b = b_vals.clone().to(torch.int64)
    for i in range(10):
        prompts[:, 20 + i] = (b % 10).to(torch.int32)
        b = b // 10
    return prompts


def _init_weights(device) -> torch.Tensor:
    w = torch.zeros(TOTAL_WEIGHTS, device=device, dtype=torch.float32)
    w[O_EMBED] = EMBED_CONST
    w[O_EMBED + 1] = DECODE_QUAD
    w[O_QPROJ] = math.cos(PHI)
    w[O_QPROJ + 1] = -math.sin(PHI)
    w[O_VPROJ] = -22.0 * DIGIT_SCALE
    w[O_GATE] = CARRY_ALPHA * (-94.0) / CONST_NORM
    w[O_GATE + 1] = CARRY_ALPHA * DIGIT_SCALE
    w[O_CARRY] = (100.0 / CARRY_ALPHA) * (1.0 / CONST_NORM)
    w[O_NORM] = (DECODE_CURVATURE / DECODE_QUAD) / CONST_NORM
    w[O_NORM + 1] = -(DIGIT_SCALE / 50.0)
    return w


def _unit_rms_norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + RMS_EPS)


def _forward_pass(seq: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = seq.shape
    device = seq.device

    embed_w = weights[O_EMBED : O_EMBED + 2]
    q_w = weights[O_QPROJ : O_QPROJ + 2]
    v_w = weights[O_VPROJ]
    gate_w = weights[O_GATE : O_GATE + 2]
    carry_w = weights[O_CARRY]
    norm_w = weights[O_NORM : O_NORM + 2]

    digits = torch.arange(VOCAB_SIZE, device=device, dtype=torch.float32)
    embed_table = torch.stack(
        [embed_w[0] - embed_w[1] * digits * digits, -digits], dim=-1
    )  # [10, 2]

    h = embed_table[seq.long()]  # [batch, seq_len, 2]

    # Pre-attention unit RMSNorm (no learned parameters)
    h_norm = _unit_rms_norm(h)

    # Q projection: [h0*qw0, h0*qw1]
    q = torch.stack([h_norm[..., 0] * q_w[0], h_norm[..., 0] * q_w[1]], dim=-1)

    # K projection: [h0, 0]
    k = torch.stack([h_norm[..., 0], torch.zeros_like(h_norm[..., 0])], dim=-1)

    # V projection: [h1*vw, 0]
    v = torch.stack([h_norm[..., 1] * v_w, torch.zeros_like(h_norm[..., 1])], dim=-1)

    # QK norm
    q = _unit_rms_norm(q)
    k = _unit_rms_norm(k)

    # RoPE
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = positions * OMEGA
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    q_rot = torch.stack(
        [q[..., 0] * cos_a - q[..., 1] * sin_a, q[..., 0] * sin_a + q[..., 1] * cos_a], dim=-1
    )
    k_rot = torch.stack(
        [k[..., 0] * cos_a - k[..., 1] * sin_a, k[..., 0] * sin_a + k[..., 1] * cos_a], dim=-1
    )

    # Attention: [batch, 1, seq_len, 2]
    q_rot = q_rot.unsqueeze(1)
    k_rot = k_rot.unsqueeze(1)
    v = v.unsqueeze(1)

    attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * ATTN_SCALE
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_probs, v).squeeze(1)  # [batch, seq_len, 2]

    # O projection: [0, attn[..., 0]]
    o = torch.stack([torch.zeros_like(attn_out[..., 0]), attn_out[..., 0]], dim=-1)

    # Residual
    h = h + o

    # Pre-MLP unit RMSNorm
    h_norm2 = _unit_rms_norm(h)

    # MLP gate projection
    a_gate = gate_w[0]
    c_gate = gate_w[1]
    g0 = h_norm2[..., 0] * a_gate + h_norm2[..., 1] * c_gate
    g1 = h_norm2[..., 0] * (a_gate - c_gate / EMBED_CONST) + h_norm2[..., 1] * c_gate
    gate = torch.stack([g0, g1], dim=-1)

    # MLP carry projection with SwiGLU
    base = h_norm2[..., 0]
    up = base.unsqueeze(-1).expand_as(gate)
    mix = F.silu(gate) * up
    mlp_out = torch.stack([torch.zeros_like(base), carry_w * (mix[..., 1] - mix[..., 0])], dim=-1)

    # Residual
    h = h + mlp_out

    # Final RMSNorm (with learned weight)
    rms = torch.sqrt(torch.mean(h * h, dim=-1, keepdim=True) + RMS_EPS)
    h = (h / rms) * norm_w

    # Output projection (tied with embedding)
    logits = h @ embed_table.T  # [batch, seq_len, 10]
    return logits


class Challenge(ChallengeBase):
    name = "Adder Transformer Inference"
    atol = 0.01
    rtol = 0.01
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        prompts: torch.Tensor,
        output: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
    ):
        assert prompts.shape == (batch_size, PROMPT_LEN)
        assert prompts.dtype == torch.int32
        assert output.shape == (batch_size, OUTPUT_DIGITS, VOCAB_SIZE)
        assert output.dtype == torch.float32
        assert weights.shape == (TOTAL_WEIGHTS,)
        assert weights.dtype == torch.float32

        seq = prompts.clone()
        for step in range(OUTPUT_DIGITS):
            logits = _forward_pass(seq, weights)
            last_logits = logits[:, -1, :]
            output[:, step, :] = last_logits
            next_token = last_logits.argmax(dim=-1).to(torch.int32)
            seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "prompts": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "weights": (ctypes.POINTER(ctypes.c_float), "in"),
            "batch_size": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        pairs = [(3, 5), (99, 1)]
        batch_size = len(pairs)
        prompts = torch.tensor(
            [_encode_pair(a, b) for a, b in pairs],
            device=device,
            dtype=torch.int32,
        )
        weights = _init_weights(device)
        output = torch.zeros(
            batch_size, OUTPUT_DIGITS, VOCAB_SIZE, device=device, dtype=torch.float32
        )
        return {
            "prompts": prompts,
            "output": output,
            "weights": weights,
            "batch_size": batch_size,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        device = self.device
        tests = []

        def _make_test(pairs):
            batch_size = len(pairs)
            prompts = torch.tensor(
                [_encode_pair(a, b) for a, b in pairs],
                device=device,
                dtype=torch.int32,
            )
            weights = _init_weights(device)
            output = torch.zeros(
                batch_size, OUTPUT_DIGITS, VOCAB_SIZE, device=device, dtype=torch.float32
            )
            return {
                "prompts": prompts,
                "output": output,
                "weights": weights,
                "batch_size": batch_size,
            }

        # Edge: single pair, both zero
        tests.append(_make_test([(0, 0)]))

        # Edge: single pair, max carry propagation
        tests.append(_make_test([(9999999999, 1)]))

        # Edge: small batch, simple sums
        tests.append(_make_test([(1, 2), (3, 4)]))

        # Power-of-2 batch: 16
        torch.manual_seed(42)
        tests.append(
            _make_test(
                [
                    (torch.randint(0, 10**10, (1,)).item(), torch.randint(0, 10**10, (1,)).item())
                    for _ in range(16)
                ]
            )
        )

        # Power-of-2 batch: 64
        tests.append(
            _make_test(
                [
                    (torch.randint(0, 10**10, (1,)).item(), torch.randint(0, 10**10, (1,)).item())
                    for _ in range(64)
                ]
            )
        )

        # Non-power-of-2: 30
        tests.append(
            _make_test(
                [
                    (torch.randint(0, 10**10, (1,)).item(), torch.randint(0, 10**10, (1,)).item())
                    for _ in range(30)
                ]
            )
        )

        # Non-power-of-2: 100
        tests.append(
            _make_test(
                [
                    (torch.randint(0, 10**10, (1,)).item(), torch.randint(0, 10**10, (1,)).item())
                    for _ in range(100)
                ]
            )
        )

        # Realistic: 1000
        tests.append(
            _make_test(
                [
                    (torch.randint(0, 10**10, (1,)).item(), torch.randint(0, 10**10, (1,)).item())
                    for _ in range(1000)
                ]
            )
        )

        # All zeros
        tests.append(_make_test([(0, 0)] * 8))

        # Max values
        tests.append(_make_test([(9999999999, 9999999999)] * 4))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        device = self.device
        batch_size = 100000
        torch.manual_seed(123)
        a_vals = torch.randint(0, 10**10, (batch_size,), dtype=torch.int64)
        b_vals = torch.randint(0, 10**10, (batch_size,), dtype=torch.int64)
        prompts = _encode_pairs_batch(a_vals, b_vals, device)
        weights = _init_weights(device)
        output = torch.zeros(
            batch_size, OUTPUT_DIGITS, VOCAB_SIZE, device=device, dtype=torch.float32
        )
        return {
            "prompts": prompts,
            "output": output,
            "weights": weights,
            "batch_size": batch_size,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_adder_transformer(...):
#     ...


def solve(
    prompts: torch.Tensor,
    output: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
