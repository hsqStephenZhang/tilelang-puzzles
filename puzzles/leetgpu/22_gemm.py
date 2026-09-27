r"""
LeetGPU 22: General Matrix Multiplication (GEMM)
================================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a basic General Matrix Multiplication (GEMM). Given matrix \(A\) of dimensions \(M \times K\), matrix \(B\) of dimensions \(K \times N\), input/output matrix \(C\) of dimensions \(M \times N\), and scalar multipliers \( \alpha \) and \( \beta \), compute the operation:
\[ C = \alpha \cdot (A \times B) + \beta \cdot C_{initial} \]

The input matrices \(A\), \(B\), and the initial state of \(C\) contain 16-bit floating-point numbers (FP16/`half`). All matrices are stored in row-major order. The scalars \( \alpha \) and \( \beta \) are 32-bit floats.

Implementation Requirements
---------------------------

- Use only native features (external libraries other than WMMA are not permitted).
- The `solve` function signature must remain unchanged.
- Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16.
- The final result must be stored back into matrix `C` as `half`.

Example:
--------

Input:

(Note: Input matrices A, B, C_initial are FP16 type for the problem)

Matrix \(A\) (\(M=2, K=3\)):
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0 \\
4.0 & 5.0 & 6.0
\end{bmatrix}
\]
Matrix \(B\) (\(K=3, N=2\)):
\[
\begin{bmatrix}
1.0 & 2.0 \\
3.0 & 4.0 \\
5.0 & 6.0
\end{bmatrix}
\]
Matrix \(C_{initial}\) (\(M=2, N=2\)):
\[
\begin{bmatrix}
1.0 & 1.0 \\
1.0 & 1.0
\end{bmatrix}
\]
\[\alpha = 1.0 \text{ (FP32)}\]
\[\beta = 0.0 \text{ (FP32)}\]

Output (FP16):

Matrix \(C\) (\(M=2, N=2\)):
\[
\begin{bmatrix}
22.0 & 28.0 \\
49.0 & 64.0
\end{bmatrix}
\]

Constraints
-----------

- 16 ≤ `M`, `N`, `K` ≤ 4096
- Performance is measured with `K` = 1,024, `M` = 1,024, `N` = 1,024

Run `python3 puzzles/leetgpu/22_gemm.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, RandTensor, run_challenge


class Challenge(ChallengeBase):
    name = "General Matrix Multiplication (GEMM)"
    atol = 0.05
    rtol = 0.05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        M: int,
        N: int,
        K: int,
        alpha: float,
        beta: float,
    ):
        assert A.shape == (M, K)
        assert B.shape == (K, N)
        assert C.shape == (M, N)
        A_f32 = A.view(M, K).to(torch.float32)
        B_f32 = B.view(K, N).to(torch.float32)
        C_f32 = C.view(M, N).to(torch.float32)
        matmul_result = torch.matmul(A_f32, B_f32)
        final_result = alpha * matmul_result + beta * C_f32
        C.copy_(final_result.to(torch.float16))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_uint16), "in"),
            "B": (ctypes.POINTER(ctypes.c_uint16), "in"),
            "C": (ctypes.POINTER(ctypes.c_uint16), "inout"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "alpha": (ctypes.c_float, "in"),
            "beta": (ctypes.c_float, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float16
        A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=self.device, dtype=dtype)
        B = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=self.device, dtype=dtype)
        C = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=self.device, dtype=dtype)
        return {
            "A": A,
            "B": B,
            "C": C,
            "M": 2,
            "N": 2,
            "K": 3,
            "alpha": 1.0,
            "beta": 0.0,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float16
        tests = []

        # 16x16x16_a1_b0
        tests.append(
            {
                "A": torch.empty((16, 16), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "B": torch.empty((16, 16), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.zeros((16, 16), device=self.device, dtype=dtype),
                "M": 16,
                "N": 16,
                "K": 16,
                "alpha": 1.0,
                "beta": 0.0,
            }
        )

        # 16x16x16_a1_b1
        tests.append(
            {
                "A": torch.empty((16, 16), device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                "B": torch.empty((16, 16), device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                "C": torch.empty((16, 16), device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                "M": 16,
                "N": 16,
                "K": 16,
                "alpha": 1.0,
                "beta": 1.0,
            }
        )

        # 32x16x16_a0.5_b0.5
        tests.append(
            {
                "A": torch.empty((32, 16), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "B": torch.empty((16, 16), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty((32, 16), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "M": 32,
                "N": 16,
                "K": 16,
                "alpha": 0.5,
                "beta": 0.5,
            }
        )

        # 16x32x16_a1_b1
        tests.append(
            {
                "A": torch.empty((16, 16), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "B": torch.empty((16, 32), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty((16, 32), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "M": 16,
                "N": 32,
                "K": 16,
                "alpha": 1.0,
                "beta": 1.0,
            }
        )

        # 16x16x32_a0_b1
        tests.append(
            {
                "A": torch.empty((16, 32), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "B": torch.empty((32, 16), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty((16, 16), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "M": 16,
                "N": 16,
                "K": 32,
                "alpha": 0.0,
                "beta": 1.0,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M = 1024
        N = 1024
        K = 1024
        A = RandTensor((M, K), -1.0, 1.0, dtype="float16")
        B = RandTensor((K, N), -1.0, 1.0, dtype="float16")
        C = RandTensor((M, N), -1.0, 1.0, dtype="float16")
        return {
            "A": A,
            "B": B,
            "C": C,
            "M": M,
            "N": N,
            "K": K,
            "alpha": 1.0,
            "beta": 1.0,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_gemm(...):
#     ...


def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    # TODO: launch your TileLang kernel and write the result into `C` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
