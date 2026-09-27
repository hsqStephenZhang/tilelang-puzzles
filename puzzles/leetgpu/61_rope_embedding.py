r"""
LeetGPU 61: Rotary Positional Embedding
=======================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a GPU program that computes the Rotary Positional Embedding (RoPE) for a batch of query vectors.
RoPE is a method for encoding positional information in transformer models by rotating the query and key vectors using precomputed cosine and sine components.

Mathematically, given a query vector \(x\) and corresponding cosine and sine vectors, the operation is defined as:
\[
\text{RoPE}(x) = x \odot \cos + \text{rotate\_half}(x) \odot \sin
\]

Where \(\odot\) denotes element-wise multiplication. The \(\text{rotate\_half}(x)\) operation swaps the first and second halves of the vector and negates the first half. For a vector of dimension \(d\):
\[
\text{rotate\_half}([x_1, \dots, x_{d/2}, x_{d/2+1}, \dots, x_d]) = [-x_{d/2+1}, \dots, -x_d, x_1, \dots, x_{d/2}]
\]

Implementation Requirements
---------------------------

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The input tensors `Q`, `cos`, and `sin` have shape `(M, D)`, where `M` is the number of tokens and `D` is the head dimension
- `cos` and `sin` use half-split layout: values at dimensions `j` and `j + D/2` are identical for each row
- `D` (head dimension) is guaranteed to be an even number
- The final result must be stored in the output variable with the same shape `(M, D)`

Example 1:
----------

Input:  Q   = [[1.0, 2.0, 3.0, 4.0],
[1.0, 1.0, 1.0, 1.0]]
Cos = [[1.0, 1.0, 1.0, 1.0],
[0.0, 0.0, 0.0, 0.0]]
Sin = [[0.0, 0.0, 0.0, 0.0],
[1.0, 1.0, 1.0, 1.0]]
Output: result = [[1.0, 2.0, 3.0, 4.0],
[-1.0, -1.0, 1.0, 1.0]]
(Row 0 is identity via Cos; Row 1 is rotated via Sin)

Constraints
-----------

- `Q`, `cos`, and `sin` have identical dimensions
- `D` % 2 == 0
- 1 ≤ `M`, `D` ≤ 10,000
- Performance is measured with `D` = 128, `M` = 1,048,576

Run `python3 puzzles/leetgpu/61_rope_embedding.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Rotary Positional Embedding"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        output: torch.Tensor,
        M: int,
        D: int,
    ):
        assert Q.shape == (M, D)
        assert cos.shape == (M, D)
        assert sin.shape == (M, D)
        assert output.shape == (M, D)

        # rotate_half implementation
        # Split the last dimension into two halves
        x1 = Q[..., : D // 2]
        x2 = Q[..., D // 2 :]
        # Concatenate -x2 and x1
        rotated_Q = torch.cat((-x2, x1), dim=-1)

        # RoPE calculation
        # Output = Q * Cos + rotate_half(Q) * Sin
        result = (Q * cos) + (rotated_Q * sin)

        output.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "cos": (ctypes.POINTER(ctypes.c_float), "in"),
            "sin": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def _half_split(self, table: torch.Tensor) -> torch.Tensor:
        D = table.shape[1]
        table[:, D // 2 :] = table[:, : D // 2]
        return table

    def generate_example_test(self) -> Dict[str, Any]:
        M = 1024
        D = 64
        dtype = torch.float32

        Q = torch.randn(M, D, device=self.device, dtype=dtype)
        Cos = self._half_split(torch.randn(M, D, device=self.device, dtype=dtype))
        Sin = self._half_split(torch.randn(M, D, device=self.device, dtype=dtype))
        Output = torch.zeros(M, D, device=self.device, dtype=dtype)

        return {
            "Q": Q,
            "cos": Cos,
            "sin": Sin,
            "output": Output,
            "M": M,
            "D": D,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []
        dtype = torch.float32

        # Test 1: Small input
        M = 4
        D = 4
        tests.append(
            {
                "Q": torch.randn(M, D, device=self.device, dtype=dtype),
                "cos": self._half_split(torch.randn(M, D, device=self.device, dtype=dtype)),
                "sin": self._half_split(torch.randn(M, D, device=self.device, dtype=dtype)),
                "output": torch.zeros(M, D, device=self.device, dtype=dtype),
                "M": M,
                "D": D,
            }
        )

        # Test 2: Larger input
        M = 128
        D = 64
        tests.append(
            {
                "Q": torch.randn(M, D, device=self.device, dtype=dtype),
                "cos": self._half_split(torch.randn(M, D, device=self.device, dtype=dtype)),
                "sin": self._half_split(torch.randn(M, D, device=self.device, dtype=dtype)),
                "output": torch.zeros(M, D, device=self.device, dtype=dtype),
                "M": M,
                "D": D,
            }
        )

        # zero_matrices: outputs should remain zero when inputs are zero
        tests.append(
            {
                "Q": torch.zeros((3, 6), device=self.device, dtype=dtype),
                "cos": torch.zeros((3, 6), device=self.device, dtype=dtype),
                "sin": torch.zeros((3, 6), device=self.device, dtype=dtype),
                "output": torch.zeros(3, 6, device=self.device, dtype=dtype),
                "M": 3,
                "D": 6,
            }
        )

        # minimal_dims: smallest even D that still allows rotation
        tests.append(
            {
                "Q": torch.randn((1, 2), device=self.device, dtype=dtype),
                "cos": self._half_split(torch.randn((1, 2), device=self.device, dtype=dtype)),
                "sin": self._half_split(torch.randn((1, 2), device=self.device, dtype=dtype)),
                "output": torch.zeros(1, 2, device=self.device, dtype=dtype),
                "M": 1,
                "D": 2,
            }
        )

        # mixed_values: negative and positive entries
        tests.append(
            {
                "Q": torch.tensor(
                    [[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "cos": torch.tensor(
                    [[0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.1, 0.2]],
                    device=self.device,
                    dtype=dtype,
                ),
                "sin": torch.tensor(
                    [[0.5, -0.5, 0.5, -0.5], [0.4, -0.3, 0.4, -0.3]],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.zeros(2, 4, device=self.device, dtype=dtype),
                "M": 2,
                "D": 4,
            }
        )

        # large_matrices: random uniform values for stress testing
        tests.append(
            {
                "Q": torch.empty((256, 128), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "cos": self._half_split(
                    torch.empty((256, 128), device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
                ),
                "sin": self._half_split(
                    torch.empty((256, 128), device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
                ),
                "output": torch.zeros(256, 128, device=self.device, dtype=dtype),
                "M": 256,
                "D": 128,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M = 1024 * 1024  # 1M tokens
        D = 128
        dtype = torch.float32
        return {
            "Q": torch.randn(M, D, device=self.device, dtype=dtype),
            "cos": self._half_split(torch.randn(M, D, device=self.device, dtype=dtype)),
            "sin": self._half_split(torch.randn(M, D, device=self.device, dtype=dtype)),
            "output": torch.zeros(M, D, device=self.device, dtype=dtype),
            "M": M,
            "D": D,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_rope_embedding(...):
#     ...


def solve(
    Q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    output: torch.Tensor,
    M: int,
    D: int,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
