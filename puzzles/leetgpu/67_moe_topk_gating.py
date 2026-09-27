r"""
LeetGPU 67: MoE Top-K Gating
============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a GPU program that performs Top-K Gating for Mixture of Experts (MoE) models. Given a logit matrix of shape `[M, E]` where M is the number of tokens and E is the number of experts, identify the k largest values in each row, extract their indices, and apply softmax to get mixing weights.

For each row i, the operation computes:
\[
\begin{align}
\text{indices}_i, \text{vals}_i &= \text{TopK}(\text{logits}_i, k) \\
\text{vals}_i &= \text{logits}_i[\text{indices}_i] \\
\text{weights}_i &= \text{Softmax}(\text{vals}_i)
\end{align}
\]

The selected experts must remain ordered by descending logit value, matching the order returned by
`topk`. The `topk_weights` array must correspond positionally to
`topk_indices` in that same order.

Implementation Requirements
---------------------------

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in the `topk_weights` and `topk_indices` arrays

Example 1:
----------

Input:
logits = [[1.0, 2.0, 3.0, 4.0],
[4.0, 3.0, 2.0, 1.0]]
M = 2, E = 4, k = 2

Output:
topk_weights = [[0.7311, 0.2689],
[0.7311, 0.2689]]
topk_indices = [[3, 2],
[0, 1]]

Explanation:
Row 0: Top-2 values are 4.0 and 3.0 at indices 3 and 2.
Softmax([4.0, 3.0]) = [0.7311, 0.2689]
Row 1: Top-2 values are 4.0 and 3.0 at indices 0 and 1.
Softmax([4.0, 3.0]) = [0.7311, 0.2689]

Constraints
-----------

- 1 ≤ `M` ≤ 10,000 (number of tokens)
- 1 ≤ `E` ≤ 256 (number of experts)
- 1 ≤ `k` ≤ `E` (top-k selection, typically k=2)
- All tensors are stored on GPU
- Logits are 32-bit floats
- Indices are 32-bit integers
- Performance is measured with `M` = 1,024, `k` = 2

Run `python3 puzzles/leetgpu/67_moe_topk_gating.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "MoE Top-K Gating"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        logits: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        M: int,
        E: int,
        k: int,
    ):
        """
        Computes the Top-K gating for Mixture of Experts.

        For each row in logits, select the k highest values, apply softmax to them,
        and return the weights and indices.
        """
        assert logits.shape == (M, E)
        assert topk_weights.shape == (M, k)
        assert topk_indices.shape == (M, k)
        assert topk_indices.dtype == torch.int32

        # 1. TopK Selection
        # logits: (M, E) -> vals: (M, k), indices: (M, k)
        vals, indices = torch.topk(logits, k, dim=-1)

        # 2. Softmax on the top k values
        weights = torch.softmax(vals, dim=-1)

        # 3. Write output
        topk_weights.copy_(weights)
        topk_indices.copy_(indices.to(torch.int32))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "topk_weights": (ctypes.POINTER(ctypes.c_float), "out"),
            "topk_indices": (ctypes.POINTER(ctypes.c_int), "out"),
            "M": (ctypes.c_int, "in"),
            "E": (ctypes.c_int, "in"),
            "k": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype_float = torch.float32
        dtype_int = torch.int32
        M = 2
        E = 4
        k = 2

        # Example from problem description
        logits_data = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], device=self.device, dtype=dtype_float
        )
        topk_weights_data = torch.zeros((M, k), device=self.device, dtype=dtype_float)
        topk_indices_data = torch.zeros((M, k), device=self.device, dtype=dtype_int)

        return {
            "logits": logits_data,
            "topk_weights": topk_weights_data,
            "topk_indices": topk_indices_data,
            "M": M,
            "E": E,
            "k": k,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype_float = torch.float32
        dtype_int = torch.int32
        test_cases = []

        # Test case 1: Basic example from problem description
        test_cases.append(
            {
                "logits": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
                    device=self.device,
                    dtype=dtype_float,
                ),
                "topk_weights": torch.zeros((2, 2), device=self.device, dtype=dtype_float),
                "topk_indices": torch.zeros((2, 2), device=self.device, dtype=dtype_int),
                "M": 2,
                "E": 4,
                "k": 2,
            }
        )

        # Test case 2: k=1 (single expert per token)
        test_cases.append(
            {
                "logits": torch.tensor(
                    [[5.0, 1.0, 3.0], [2.0, 8.0, 4.0], [6.0, 2.0, 9.0]],
                    device=self.device,
                    dtype=dtype_float,
                ),
                "topk_weights": torch.zeros((3, 1), device=self.device, dtype=dtype_float),
                "topk_indices": torch.zeros((3, 1), device=self.device, dtype=dtype_int),
                "M": 3,
                "E": 3,
                "k": 1,
            }
        )

        # Test case 3: k=E (all experts)
        test_cases.append(
            {
                "logits": torch.tensor(
                    [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], device=self.device, dtype=dtype_float
                ),
                "topk_weights": torch.zeros((2, 3), device=self.device, dtype=dtype_float),
                "topk_indices": torch.zeros((2, 3), device=self.device, dtype=dtype_int),
                "M": 2,
                "E": 3,
                "k": 3,
            }
        )

        # Test case 4: Typical MoE configuration (M=4, E=8, k=2)
        torch.manual_seed(42)
        test_cases.append(
            {
                "logits": torch.randn((4, 8), device=self.device, dtype=dtype_float),
                "topk_weights": torch.zeros((4, 2), device=self.device, dtype=dtype_float),
                "topk_indices": torch.zeros((4, 2), device=self.device, dtype=dtype_int),
                "M": 4,
                "E": 8,
                "k": 2,
            }
        )

        # Test case 5: Larger E with small k (M=8, E=64, k=2)
        torch.manual_seed(123)
        test_cases.append(
            {
                "logits": torch.randn((8, 64), device=self.device, dtype=dtype_float),
                "topk_weights": torch.zeros((8, 2), device=self.device, dtype=dtype_float),
                "topk_indices": torch.zeros((8, 2), device=self.device, dtype=dtype_int),
                "M": 8,
                "E": 64,
                "k": 2,
            }
        )

        # Test case 6: Test with negative logits
        test_cases.append(
            {
                "logits": torch.tensor(
                    [[-1.0, -2.0, -3.0, -4.0], [-4.0, -1.0, -2.0, -3.0]],
                    device=self.device,
                    dtype=dtype_float,
                ),
                "topk_weights": torch.zeros((2, 2), device=self.device, dtype=dtype_float),
                "topk_indices": torch.zeros((2, 2), device=self.device, dtype=dtype_int),
                "M": 2,
                "E": 4,
                "k": 2,
            }
        )

        # Test case 7: Medium size test (M=100, E=16, k=4)
        torch.manual_seed(456)
        test_cases.append(
            {
                "logits": torch.randn((100, 16), device=self.device, dtype=dtype_float),
                "topk_weights": torch.zeros((100, 4), device=self.device, dtype=dtype_float),
                "topk_indices": torch.zeros((100, 4), device=self.device, dtype=dtype_int),
                "M": 100,
                "E": 16,
                "k": 4,
            }
        )

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype_float = torch.float32
        dtype_int = torch.int32
        M = 1024
        E = 64
        k = 2

        torch.manual_seed(789)
        return {
            "logits": torch.randn((M, E), device=self.device, dtype=dtype_float),
            "topk_weights": torch.zeros((M, k), device=self.device, dtype=dtype_float),
            "topk_indices": torch.zeros((M, k), device=self.device, dtype=dtype_int),
            "M": M,
            "E": E,
            "k": k,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_moe_topk_gating(...):
#     ...


def solve(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    M: int,
    E: int,
    k: int,
):
    # TODO: launch your TileLang kernel and write the result into `topk_weights`, `topk_indices` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
