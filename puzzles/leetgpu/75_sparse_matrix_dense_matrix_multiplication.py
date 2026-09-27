r"""
LeetGPU 75: Sparse Matrix-Dense Matrix Multiplication
=====================================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a GPU program that multiplies a sparse matrix `A` of dimensions `M` × `N`
by a dense matrix `B` of dimensions `N` × `K`, producing a dense output matrix
`C` of dimensions `M` × `K`.
All matrices are stored in row-major order using 32-bit floats.
The matrix `A` is approximately 60–70% sparse (i.e., 60–70% of elements are zero),
and `nnz` gives the number of non-zero elements in `A`.

Mathematically, the operation is defined as:
\[
C_{ij} = \sum_{k=0}^{N-1} A_{ik} \cdot B_{kj} \quad \text{for} \quad i = 0, \ldots, M-1,\; j = 0, \ldots, K-1
\]

Implementation Requirements
---------------------------

- Use only GPU native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in matrix `C`

Example
-------

Input:

Matrix \(A\) (\(3 \times 4\)):
\[
\begin{bmatrix}
2.0 & 0.0 & 0.0 & 1.0 \\
0.0 & 3.0 & 0.0 & 0.0 \\
0.0 & 0.0 & 4.0 & 0.0
\end{bmatrix}
\]
Matrix \(B\) (\(4 \times 2\)):
\[
\begin{bmatrix}
1.0 & 2.0 \\
3.0 & 4.0 \\
5.0 & 6.0 \\
7.0 & 8.0
\end{bmatrix}
\]
Output:

Matrix \(C\) (\(3 \times 2\)):
\[
\begin{bmatrix}
9.0 & 12.0 \\
9.0 & 12.0 \\
20.0 & 24.0
\end{bmatrix}
\]

Constraints
-----------

- 1 ≤ `M`, `N`, `K` ≤ 8,192
- All values in `A` and `B` are 32-bit floats in the range [−10, 10]
- The matrix `A` is approximately 60–70% sparse
- Performance is measured with `M` = 4,096, `N` = 2,048, `K` = 512

Run `python3 puzzles/leetgpu/75_sparse_matrix_dense_matrix_multiplication.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Sparse Matrix-Dense Matrix Multiplication"
    atol = 0.001
    rtol = 0.001
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
        nnz: int,
    ):
        if A.shape == (M * N,):
            A_matrix = A.view(M, N)
        elif A.shape == (M, N):
            A_matrix = A
        else:
            raise AssertionError(
                f"A.shape {A.shape} does not match expected {(M * N,)} or {(M, N)}"
            )
        if B.shape == (N * K,):
            B_matrix = B.view(N, K)
        elif B.shape == (N, K):
            B_matrix = B
        else:
            raise AssertionError(
                f"B.shape {B.shape} does not match expected {(N * K,)} or {(N, K)}"
            )
        assert C.shape == (M, K) or C.shape == (
            M * K,
        ), f"C.shape {C.shape} does not match expected {(M, K)} or {(M * K,)}"
        assert A_matrix.dtype == torch.float32
        assert B_matrix.dtype == torch.float32
        result = torch.matmul(A_matrix, B_matrix)
        C.copy_(result.view(C.shape))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "nnz": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.tensor(
            [
                [2.0, 0.0, 0.0, 1.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        B = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        C = torch.empty((3, 2), device=self.device, dtype=dtype)
        return {
            "A": A,
            "B": B,
            "C": C,
            "M": 3,
            "N": 4,
            "K": 2,
            "nnz": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # edge_1x1x1
        tests.append(
            {
                "A": torch.tensor([[3.0]], device=self.device, dtype=dtype),
                "B": torch.tensor([[2.0]], device=self.device, dtype=dtype),
                "C": torch.empty((1, 1), device=self.device, dtype=dtype),
                "M": 1,
                "N": 1,
                "K": 1,
                "nnz": 1,
            }
        )

        # edge_2x2_k1_spmv_like
        tests.append(
            {
                "A": torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=self.device, dtype=dtype),
                "B": torch.tensor([[3.0], [4.0]], device=self.device, dtype=dtype),
                "C": torch.empty((2, 1), device=self.device, dtype=dtype),
                "M": 2,
                "N": 2,
                "K": 1,
                "nnz": 2,
            }
        )

        # edge_zero_matrix
        tests.append(
            {
                "A": torch.zeros((3, 3), device=self.device, dtype=dtype),
                "B": torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=self.device, dtype=dtype
                ),
                "C": torch.empty((3, 2), device=self.device, dtype=dtype),
                "M": 3,
                "N": 3,
                "K": 2,
                "nnz": 0,
            }
        )

        # edge_identity_a
        tests.append(
            {
                "A": torch.eye(4, device=self.device, dtype=dtype),
                "B": torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "C": torch.empty((4, 3), device=self.device, dtype=dtype),
                "M": 4,
                "N": 4,
                "K": 3,
                "nnz": 4,
            }
        )

        # power_of_2_16x16x8
        M, N, K = 16, 16, 8
        A_dense = torch.empty((M, N), device=self.device, dtype=dtype).uniform_(-2.0, 2.0)
        mask = torch.rand((M, N), device=self.device) > 0.65
        A_sparse = A_dense * mask
        tests.append(
            {
                "A": A_sparse,
                "B": torch.empty((N, K), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty((M, K), device=self.device, dtype=dtype),
                "M": M,
                "N": N,
                "K": K,
                "nnz": int(mask.sum().item()),
            }
        )

        # power_of_2_64x32x16
        M, N, K = 64, 32, 16
        A_dense = torch.empty((M, N), device=self.device, dtype=dtype).uniform_(-3.0, 3.0)
        mask = torch.rand((M, N), device=self.device) > 0.70
        A_sparse = A_dense * mask
        tests.append(
            {
                "A": A_sparse,
                "B": torch.empty((N, K), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty((M, K), device=self.device, dtype=dtype),
                "M": M,
                "N": N,
                "K": K,
                "nnz": int(mask.sum().item()),
            }
        )

        # non_power_of_2_negative_values
        M, N, K = 30, 50, 20
        A_dense = torch.empty((M, N), device=self.device, dtype=dtype).uniform_(-5.0, 5.0)
        mask = torch.rand((M, N), device=self.device) > 0.65
        A_sparse = A_dense * mask
        tests.append(
            {
                "A": A_sparse,
                "B": torch.empty((N, K), device=self.device, dtype=dtype).uniform_(-3.0, 3.0),
                "C": torch.empty((M, K), device=self.device, dtype=dtype),
                "M": M,
                "N": N,
                "K": K,
                "nnz": int(mask.sum().item()),
            }
        )

        # non_power_of_2_255x100x33
        M, N, K = 255, 100, 33
        A_dense = torch.empty((M, N), device=self.device, dtype=dtype).uniform_(-2.0, 2.0)
        mask = torch.rand((M, N), device=self.device) > 0.70
        A_sparse = A_dense * mask
        tests.append(
            {
                "A": A_sparse,
                "B": torch.empty((N, K), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty((M, K), device=self.device, dtype=dtype),
                "M": M,
                "N": N,
                "K": K,
                "nnz": int(mask.sum().item()),
            }
        )

        # realistic_1000x500x64
        M, N, K = 1000, 500, 64
        A_dense = torch.empty((M, N), device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        mask = torch.rand((M, N), device=self.device) > 0.65
        A_sparse = A_dense * mask
        tests.append(
            {
                "A": A_sparse,
                "B": torch.empty((N, K), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty((M, K), device=self.device, dtype=dtype),
                "M": M,
                "N": N,
                "K": K,
                "nnz": int(mask.sum().item()),
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M = 4096
        N = 2048
        K = 512
        A_dense = torch.empty((M, N), device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        mask = torch.rand((M, N), device=self.device) > 0.65
        A_sparse = A_dense * mask
        nnz = int(mask.sum().item())
        B = torch.empty((N, K), device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        C = torch.empty((M, K), device=self.device, dtype=dtype)
        return {
            "A": A_sparse,
            "B": B,
            "C": C,
            "M": M,
            "N": N,
            "K": K,
            "nnz": nnz,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_sparse_matrix_dense_matrix_multiplication(...):
#     ...


def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    nnz: int,
):
    # TODO: launch your TileLang kernel and write the result into `C` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
