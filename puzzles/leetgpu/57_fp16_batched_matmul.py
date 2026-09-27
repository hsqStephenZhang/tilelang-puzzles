r"""
LeetGPU 57: FP16 Batched Matrix Multiplication
==============================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a batched matrix multiplication in FP16. Given a batch of matrices `A` of shape `[B, M, K]` and a batch of matrices `B` of shape `[B, K, N]`, compute the output batch `C` of shape `[B, M, N]` such that for each batch index `b`:
\[
C_b = A_b \times B_b
\]
All matrices are stored in row-major order and use 16-bit floating point numbers (FP16/`half`). Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16.

Implementation Requirements
---------------------------

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16
- The final result must be stored in the `C` array as `half`

Example 1:
----------

Input:
B = 2, M = 2, K = 3, N = 2
A = [
[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
]
B = [
[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
[[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]
]
Output:
C = [
[[22.0, 28.0], [49.0, 64.0]],
[[92.0, 68.0], [128.0, 95.0]]
]

Constraints
-----------

- 1 ≤ `B` ≤ 128
- 1 ≤ `M`, `N`, `K` ≤ 1024
- Performance is measured with `K` = 256, `M` = 256, `N` = 256

Run `python3 puzzles/leetgpu/57_fp16_batched_matmul.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "FP16 Batched Matrix Multiplication"
    atol = 0.05
    rtol = 0.05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, BATCH: int, M: int, N: int, K: int
    ):
        # A: (BATCH, M, K), B: (BATCH, K, N), C: (BATCH, M, N)
        A = A.view(BATCH, M, K)
        B = B.view(BATCH, K, N)
        # Use FP32 for accumulation, then convert to FP16
        A_f32 = A.to(torch.float32)
        B_f32 = B.to(torch.float32)
        result = torch.bmm(A_f32, B_f32)
        C.copy_(result.to(torch.float16))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_uint16), "in"),
            "B": (ctypes.POINTER(ctypes.c_uint16), "in"),
            "C": (ctypes.POINTER(ctypes.c_uint16), "out"),
            "BATCH": (ctypes.c_int, "in"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float16
        BATCH, M, K, N = 2, 2, 3, 2
        A = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
            device=self.device,
            dtype=dtype,
        )
        B = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]],
            device=self.device,
            dtype=dtype,
        )
        C = torch.empty(BATCH, M, N, device=self.device, dtype=dtype)
        return {"A": A, "B": B, "C": C, "BATCH": BATCH, "M": M, "N": N, "K": K}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float16
        device = self.device
        tests = []

        # 1. basic_example
        A1 = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
            device=device,
            dtype=dtype,
        ).flatten()
        B1 = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]],
            device=device,
            dtype=dtype,
        ).flatten()
        C1 = torch.empty((2, 2, 2), device=device, dtype=dtype)
        tests.append({"A": A1, "B": B1, "C": C1, "BATCH": 2, "M": 2, "N": 2, "K": 3})

        # 2. single_batch
        A2 = torch.tensor(
            [[[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]], device=device, dtype=dtype
        ).flatten()
        B2 = torch.tensor(
            [[[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 1.0, 2.0]]], device=device, dtype=dtype
        ).flatten()
        C2 = torch.empty((1, 3, 3), device=device, dtype=dtype)
        tests.append({"A": A2, "B": B2, "C": C2, "BATCH": 1, "M": 3, "N": 3, "K": 3})

        # 3. batch_4_small
        A3 = torch.empty((4, 2, 2), device=device, dtype=dtype).uniform_(-1.0, 1.0)
        B3 = torch.empty((4, 2, 2), device=device, dtype=dtype).uniform_(-1.0, 1.0)
        C3 = torch.empty((4, 2, 2), device=device, dtype=dtype)
        tests.append({"A": A3, "B": B3, "C": C3, "BATCH": 4, "M": 2, "N": 2, "K": 2})

        # 4. batch_8_rectangular
        A4 = torch.empty((8, 4, 2), device=device, dtype=dtype).uniform_(-5.0, 5.0)
        B4 = torch.empty((8, 2, 3), device=device, dtype=dtype).uniform_(-5.0, 5.0)
        C4 = torch.empty((8, 4, 3), device=device, dtype=dtype)
        tests.append({"A": A4, "B": B4, "C": C4, "BATCH": 8, "M": 4, "N": 3, "K": 2})

        # 5. batch_16_medium
        A5 = torch.empty((16, 16, 16), device=device, dtype=dtype).uniform_(-1.0, 1.0)
        B5 = torch.empty((16, 16, 16), device=device, dtype=dtype).uniform_(-1.0, 1.0)
        C5 = torch.empty((16, 16, 16), device=device, dtype=dtype)
        tests.append({"A": A5, "B": B5, "C": C5, "BATCH": 16, "M": 16, "N": 16, "K": 16})

        # 6. batch_2_non_square
        A6 = torch.empty((2, 8, 4), device=device, dtype=dtype).uniform_(-5.0, 5.0)
        B6 = torch.empty((2, 4, 6), device=device, dtype=dtype).uniform_(-5.0, 5.0)
        C6 = torch.empty((2, 8, 6), device=device, dtype=dtype)
        tests.append({"A": A6, "B": B6, "C": C6, "BATCH": 2, "M": 8, "N": 6, "K": 4})

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float16
        BATCH, M, N, K = 32, 256, 256, 256
        A = torch.empty(BATCH, M, K, device=self.device, dtype=dtype).uniform_(-5.0, 5.0)
        B = torch.empty(BATCH, K, N, device=self.device, dtype=dtype).uniform_(-5.0, 5.0)
        C = torch.empty(BATCH, M, N, device=self.device, dtype=dtype)
        return {"A": A, "B": B, "C": C, "BATCH": BATCH, "M": M, "N": N, "K": K}


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_fp16_batched_matmul(...):
#     ...


def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BATCH: int,
    M: int,
    N: int,
    K: int,
):
    # TODO: launch your TileLang kernel and write the result into `C` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
