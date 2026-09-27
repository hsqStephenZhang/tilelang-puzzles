r"""
LeetGPU 90: Causal Depthwise Conv1d
===================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a causal depthwise 1D convolution over a batched sequence tensor
`x` of shape `(B, L, D)`, producing an output of the same shape.
In a depthwise convolution, each channel `d` is convolved independently using its
own kernel `weight[d, :]` — there is no mixing across channels.
The convolution is causal: output position `l` may only depend on
input positions `0, 1, …, l` (past and present), never future positions.
This operation is a key component of state-space models such as Mamba, where it is applied
before the selective scan to mix local context within each feature channel.

(diagram omitted, see challenge.html in leetgpu-challenges)

Formally, for each batch element `b`, sequence position `l`, and channel `d`:

\[
\text{output}[b,\, l,\, d]
= \text{bias}[d]
+ \sum_{k=0}^{K-1} \text{weight}[d,\, k] \cdot x[b,\, l - k,\, d]
\]

where positions `l − k < 0` are treated as zero (zero-pad the left boundary).
The tensor layout is channels-last: `x[b, l, d]` is stored at offset
`b × L × D + l × D + d`.

Implementation Requirements
---------------------------

- The `solve` function signature must remain unchanged
- The result must be written into the `output` tensor
- Use only native features (external libraries are not permitted)
- Input positions before the start of the sequence (i.e. indices `l − k < 0`) must be treated as zero

Example
-------

With `B` = 1, `L` = 4, `D` = 2, `K` = 3:

x      = [[[1.0, 2.0],    # l=0
[3.0, 4.0],    # l=1
[5.0, 6.0],    # l=2
[7.0, 8.0]]]   # l=3   shape (1, 4, 2)

weight = [[ 1.0,  0.0, -1.0],   # channel d=0
[ 1.0,  1.0,  1.0]]   # channel d=1   shape (2, 3)

bias   = [0.0, 0.0]

output = [[[1.0,  2.0],   # l=0: d0: 1*1=1          d1: 1*2=2
[3.0,  6.0],   # l=1: d0: 3*1+1*0=3      d1: 4*1+2*1=6
[4.0, 12.0],   # l=2: d0: 5*1+3*0+1*(-1)=4  d1: 6+4+2=12
[4.0, 18.0]]]  # l=3: d0: 7*1+5*0+3*(-1)=4  d1: 8+6+4=18

Constraints
-----------

- 1 ≤ `B` ≤ 16 (batch size)
- 1 ≤ `L` ≤ 8,192 (sequence length)
- 1 ≤ `D` ≤ 8,192 (number of channels)
- 1 ≤ `K` ≤ 8 (kernel size; typically 3 or 4 in practice)
- All tensors use 32-bit floating point
- Tensor `x` and `output` use channels-last layout: shape `(B, L, D)`
- Performance is measured with `B` = 8, `L` = 2,048, `D` = 4,096, `K` = 4

Run `python3 puzzles/leetgpu/90_causal_depthwise_conv1d.py` to test your `solve`, or add `--ref` to run the
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
    name = "Causal Depthwise Conv1d"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
        B: int,
        L: int,
        D: int,
        K: int,
    ):
        assert x.shape == (B, L, D)
        assert weight.shape == (D, K)
        assert bias.shape == (D,)
        assert output.shape == (B, L, D)
        assert x.dtype == weight.dtype == bias.dtype == output.dtype == torch.float32

        # Reshape to (B, D, L) for conv1d
        x_t = x.permute(0, 2, 1).contiguous()  # (B, D, L)

        # Causal padding: pad K-1 zeros on the left so each output position
        # only sees current and past input positions
        x_padded = F.pad(x_t, (K - 1, 0))  # (B, D, L + K - 1)

        # Depthwise conv: weight (D, K) -> (D, 1, K), groups=D
        # Flip the kernel so weight[d, 0] applies to the current position (l-0)
        # and weight[d, K-1] applies to the oldest position (l-(K-1)).
        # F.conv1d uses cross-correlation (no implicit flip), so we flip explicitly.
        w = weight.flip(1).unsqueeze(1)  # (D, 1, K)
        result = F.conv1d(x_padded, w, bias=bias, groups=D)  # (B, D, L)

        output.copy_(result.permute(0, 2, 1))  # (B, L, D)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "bias": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "L": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        B, L, D, K = 1, 4, 2, 3
        x = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
            device=self.device,
            dtype=torch.float32,
        )
        weight = torch.tensor(
            [[1.0, 0.0, -1.0], [1.0, 1.0, 1.0]], device=self.device, dtype=torch.float32
        )
        bias = torch.zeros(D, device=self.device, dtype=torch.float32)
        output = torch.empty(B, L, D, device=self.device, dtype=torch.float32)
        return {
            "x": x,
            "weight": weight,
            "bias": bias,
            "output": output,
            "B": B,
            "L": L,
            "D": D,
            "K": K,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []

        def make_case(B, L, D, K, x_vals=None, w_vals=None, b_vals=None):
            if x_vals is not None:
                x = torch.tensor(x_vals, device=self.device, dtype=dtype)
            else:
                x = torch.randn(B, L, D, device=self.device, dtype=dtype)
            if w_vals is not None:
                weight = torch.tensor(w_vals, device=self.device, dtype=dtype)
            else:
                weight = torch.randn(D, K, device=self.device, dtype=dtype)
            if b_vals is not None:
                bias = torch.tensor(b_vals, device=self.device, dtype=dtype)
            else:
                bias = torch.randn(D, device=self.device, dtype=dtype)
            output = torch.empty(B, L, D, device=self.device, dtype=dtype)
            return {
                "x": x,
                "weight": weight,
                "bias": bias,
                "output": output,
                "B": B,
                "L": L,
                "D": D,
                "K": K,
            }

        # Example test (matches generate_example_test)
        test_cases.append(
            make_case(
                1,
                4,
                2,
                3,
                x_vals=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
                w_vals=[[1.0, 0.0, -1.0], [1.0, 1.0, 1.0]],
                b_vals=[0.0, 0.0],
            )
        )

        # Edge cases: minimal sizes
        test_cases.append(make_case(1, 1, 1, 1))  # single element, kernel=1
        test_cases.append(make_case(1, 2, 1, 2))  # L < K, so first output is partial
        test_cases.append(make_case(2, 3, 4, 3))  # small batch, B=2

        # Zero inputs
        x_zero = torch.zeros(1, 8, 4, device=self.device, dtype=dtype)
        w_zero = torch.randn(4, 3, device=self.device, dtype=dtype)
        b_zero = torch.randn(4, device=self.device, dtype=dtype)
        test_cases.append(
            {
                "x": x_zero,
                "weight": w_zero,
                "bias": b_zero,
                "output": torch.empty(1, 8, 4, device=self.device, dtype=dtype),
                "B": 1,
                "L": 8,
                "D": 4,
                "K": 3,
            }
        )

        # Negative values
        test_cases.append(make_case(1, 16, 8, 4))

        # Power-of-2 sizes
        test_cases.append(make_case(2, 32, 16, 4))
        test_cases.append(make_case(4, 64, 32, 4))

        # Non-power-of-2 sizes
        test_cases.append(make_case(3, 30, 12, 3))
        test_cases.append(make_case(2, 100, 24, 4))

        # Realistic inference size (Mamba-like small)
        test_cases.append(make_case(2, 256, 128, 4))

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        B, L, D, K = 8, 2048, 4096, 4
        return {
            "x": RandnTensor((B, L, D)),
            "weight": RandnTensor((D, K)),
            "bias": RandnTensor((D,)),
            "output": OutTensor((B, L, D)),
            "B": B,
            "L": L,
            "D": D,
            "K": K,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_causal_depthwise_conv1d(...):
#     ...


def solve(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    B: int,
    L: int,
    D: int,
    K: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
