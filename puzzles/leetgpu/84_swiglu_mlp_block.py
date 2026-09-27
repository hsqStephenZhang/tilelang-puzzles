r"""
LeetGPU 84: SwiGLU MLP Block
============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement the SwiGLU MLP block — the feedforward network used in LLaMA, Mistral, Gemma, and most
modern large language models. Given an input matrix `x` of shape
`[M, d_model]` and three weight matrices `W_gate`, `W_up`
(each `[d_model, d_ffn]`), and `W_down` (`[d_ffn, d_model]`),
compute:
`output = (SiLU(x × W_gate) ⊙ (x × W_up)) × W_down`,
where `SiLU(z) = z × sigmoid(z)` and `⊙` denotes element-wise
multiplication. All tensors are `float32`.

(diagram omitted, see challenge.html in leetgpu-challenges)

Implementation Requirements
---------------------------

- Implement the `solve` function with the signature unchanged.
- Do not use external libraries beyond the framework provided.
- Write the result into `output` in-place.

Example
-------

Input: `M` = 2, `d_model` = 2, `d_ffn` = 4

\(x\) (float32, \(2 \times 2\)):
\[
x = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}
\]
\(W_\text{gate}\) and \(W_\text{up}\) (both \(2 \times 4\)):
\[
W_\text{gate} = W_\text{up} =
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0
\end{bmatrix}
\]
\(W_\text{down}\) (\(4 \times 2\)):
\[
W_\text{down} =
\begin{bmatrix}
1.0 & 0.0 \\
0.0 & 1.0 \\
0.0 & 0.0 \\
0.0 & 0.0
\end{bmatrix}
\]

Intermediate steps:
\[
\text{gate} = x \cdot W_\text{gate} =
\begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \end{bmatrix}
\]
\[
\text{up} = x \cdot W_\text{up} =
\begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \end{bmatrix}
\]
\[
\text{SiLU}(1.0) = 1.0 \times \sigma(1.0) \approx 0.7311
\]
\[
\text{hidden} = \text{SiLU}(\text{gate}) \odot \text{up} =
\begin{bmatrix} 0.7311 & 0.0 & 0.0 & 0.0 \\ 0.0 & 0.7311 & 0.0 & 0.0 \end{bmatrix}
\]

Output:
\[
\text{output} = \text{hidden} \cdot W_\text{down} \approx
\begin{bmatrix} 0.7311 & 0.0 \\ 0.0 & 0.7311 \end{bmatrix}
\]

Constraints
-----------

- 1 ≤ `M` ≤ 65,536
- 1 ≤ `d_model` ≤ 8,192
- 1 ≤ `d_ffn` ≤ 32,768
- All tensors are `float32` on the GPU.
- Input values are in the range [-10, 10].

-
Performance is measured with `M` = 512, `d_model` = 4,096,
`d_ffn` = 14,336

Run `python3 puzzles/leetgpu/84_swiglu_mlp_block.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
import torch.nn.functional as F
from common.leetgpu import ChallengeBase, OutTensor, RandnTensor, run_challenge


class Challenge(ChallengeBase):
    name = "SwiGLU MLP Block"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        W_gate: torch.Tensor,
        W_up: torch.Tensor,
        W_down: torch.Tensor,
        output: torch.Tensor,
        M: int,
        d_model: int,
        d_ffn: int,
    ):
        assert x.shape == (M, d_model)
        assert W_gate.shape == (d_model, d_ffn)
        assert W_up.shape == (d_model, d_ffn)
        assert W_down.shape == (d_ffn, d_model)
        assert output.shape == (M, d_model)
        assert (
            x.dtype == W_gate.dtype == W_up.dtype == W_down.dtype == output.dtype == torch.float32
        )

        gate = x @ W_gate  # [M, d_ffn]
        up = x @ W_up  # [M, d_ffn]
        hidden = F.silu(gate) * up  # [M, d_ffn]
        output.copy_(hidden @ W_down)  # [M, d_model]

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_gate": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_up": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_down": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "d_ffn": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, M, d_model, d_ffn, zero_x=False):
        device = self.device
        dtype = torch.float32
        if zero_x:
            x = torch.zeros(M, d_model, device=device, dtype=dtype)
        else:
            x = torch.randn(M, d_model, device=device, dtype=dtype) * 0.1
        W_gate = torch.randn(d_model, d_ffn, device=device, dtype=dtype) * 0.02
        W_up = torch.randn(d_model, d_ffn, device=device, dtype=dtype) * 0.02
        W_down = torch.randn(d_ffn, d_model, device=device, dtype=dtype) * 0.02
        output = torch.empty(M, d_model, device=device, dtype=dtype)
        return {
            "x": x,
            "W_gate": W_gate,
            "W_up": W_up,
            "W_down": W_down,
            "output": output,
            "M": M,
            "d_model": d_model,
            "d_ffn": d_ffn,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        M, d_model, d_ffn = 2, 2, 4
        # x: each row is a basis vector
        x = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        # W_gate: [d_model=2, d_ffn=4] — first two columns are identity, rest zeros
        W_gate = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        # W_up: same layout as W_gate
        W_up = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        # W_down: [d_ffn=4, d_model=2] — top 2x2 is identity, rest zeros
        W_down = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        output = torch.empty(M, d_model, device=device, dtype=dtype)
        return {
            "x": x,
            "W_gate": W_gate,
            "W_up": W_up,
            "W_down": W_down,
            "output": output,
            "M": M,
            "d_model": d_model,
            "d_ffn": d_ffn,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge cases: single row
        tests.append(self._make_test_case(1, 4, 8))

        # Edge case: two rows
        tests.append(self._make_test_case(2, 4, 8))

        # Zero input
        tests.append(self._make_test_case(4, 8, 16, zero_x=True))

        # Power-of-2 sizes
        tests.append(self._make_test_case(16, 32, 64))

        # Power-of-2 larger
        tests.append(self._make_test_case(64, 64, 128))

        # Non-power-of-2 M
        tests.append(self._make_test_case(30, 32, 64))

        # Non-power-of-2 all dims
        tests.append(self._make_test_case(100, 60, 120))

        # Non-power-of-2 M, medium size
        tests.append(self._make_test_case(255, 64, 128))

        # Realistic small inference batch (LLaMA-style ratios)
        tests.append(self._make_test_case(128, 256, 512))

        # Realistic medium inference batch
        tests.append(self._make_test_case(256, 512, 1024))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # LLaMA-3 8B style: d_model=4096, d_ffn=14336, M=512 (batch=4 x seq=128)
        M, d_model, d_ffn = 512, 4096, 14336
        return {
            "x": RandnTensor((M, d_model), std=0.1),
            "W_gate": RandnTensor((d_model, d_ffn), std=0.02),
            "W_up": RandnTensor((d_model, d_ffn), std=0.02),
            "W_down": RandnTensor((d_ffn, d_model), std=0.02),
            "output": OutTensor((M, d_model)),
            "M": M,
            "d_model": d_model,
            "d_ffn": d_ffn,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_swiglu_mlp_block(...):
#     ...


def solve(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d_model: int,
    d_ffn: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
