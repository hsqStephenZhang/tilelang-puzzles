r"""
LeetGPU 40: Batch Normalization
===============================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement batch normalization forward pass for 2D input tensors. Given an input tensor of shape [N, C] where N is the batch size and C is the number of features, compute the normalized output using learnable scale (`gamma`) and shift (`beta`) parameters.

For each feature channel j, batch normalization computes:
\[
\begin{align}
\mu_j &= \frac{1}{N} \sum_{i=1}^{N} x_{i,j} \\
\sigma_j^2 &= \frac{1}{N} \sum_{i=1}^{N} (x_{i,j} - \mu_j)^2 \\
\hat{x}_{i,j} &= \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} \\
y_{i,j} &= \gamma_j \hat{x}_{i,j} + \beta_j
\end{align}
\]

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` tensor

Example 1:
----------

Input:  input = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  (N=3, C=2)
gamma = [1.0, 1.0]
beta = [0.0, 0.0]
eps = 1e-5
Output: output = [[-1.224, -1.224], [0.0, 0.0], [1.224, 1.224]]

Example 2:
----------

Input:  input = [[0.0, 1.0], [2.0, 3.0]]  (N=2, C=2)
gamma = [2.0, 0.5]
beta = [1.0, -1.0]
eps = 1e-5
Output: output = [[-1.0, -1.5], [3.0, -0.5]]

Constraints
-----------

- 1 ≤ `N` ≤ 10,000
- 1 ≤ `C` ≤ 1,024
- `eps` = 1e-5
- -100.0 ≤ input values ≤ 100.0
- 0.1 ≤ gamma values ≤ 10.0
- -10.0 ≤ beta values ≤ 10.0
- Performance is measured with `N` = 5,000

Run `python3 puzzles/leetgpu/40_batch_normalization.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Batch Normalization"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        input: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        output: torch.Tensor,
        N: int,
        C: int,
        eps: float,
    ):
        assert input.shape == output.shape == (N, C)
        assert gamma.shape == beta.shape == (C,)
        assert input.dtype == gamma.dtype == beta.dtype == output.dtype
        assert input.device == gamma.device == beta.device == output.device

        # Compute mean and variance for each feature channel
        mean = torch.mean(input, dim=0)  # Shape: [C]
        variance = torch.var(input, dim=0, unbiased=False)  # Shape: [C]

        # Normalize
        normalized = (input - mean) / torch.sqrt(variance + eps)

        # Scale and shift
        output.copy_(gamma * normalized + beta)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "beta": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N, C = 3, 2
        input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=self.device, dtype=dtype)
        gamma = torch.tensor([1.0, 1.0], device=self.device, dtype=dtype)
        beta = torch.tensor([0.0, 0.0], device=self.device, dtype=dtype)
        output = torch.empty((N, C), device=self.device, dtype=dtype)
        eps = 1e-5
        return {
            "input": input,
            "gamma": gamma,
            "beta": beta,
            "output": output,
            "N": N,
            "C": C,
            "eps": eps,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_small
        N, C = 3, 2
        tests.append(
            {
                "input": torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=self.device, dtype=dtype
                ),
                "gamma": torch.tensor([1.0, 1.0], device=self.device, dtype=dtype),
                "beta": torch.tensor([0.0, 0.0], device=self.device, dtype=dtype),
                "output": torch.empty((N, C), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # single_batch
        N, C = 1, 4
        tests.append(
            {
                "input": torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=self.device, dtype=dtype),
                "gamma": torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device, dtype=dtype),
                "beta": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype),
                "output": torch.empty((N, C), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # all_zeros
        N, C = 4, 3
        tests.append(
            {
                "input": torch.zeros((N, C), device=self.device, dtype=dtype),
                "gamma": torch.ones(C, device=self.device, dtype=dtype),
                "beta": torch.zeros(C, device=self.device, dtype=dtype),
                "output": torch.empty((N, C), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # negative_numbers
        N, C = 2, 3
        tests.append(
            {
                "input": torch.tensor(
                    [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], device=self.device, dtype=dtype
                ),
                "gamma": torch.tensor([1.0, 1.0, 1.0], device=self.device, dtype=dtype),
                "beta": torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=dtype),
                "output": torch.empty((N, C), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # different_gamma_beta
        N, C = 2, 2
        tests.append(
            {
                "input": torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=self.device, dtype=dtype),
                "gamma": torch.tensor([2.0, 0.5], device=self.device, dtype=dtype),
                "beta": torch.tensor([1.0, -1.0], device=self.device, dtype=dtype),
                "output": torch.empty((N, C), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # large_values
        N, C = 5, 3
        tests.append(
            {
                "input": torch.empty((N, C), device=self.device, dtype=dtype).uniform_(-50.0, 50.0),
                "gamma": torch.empty(C, device=self.device, dtype=dtype).uniform_(0.5, 2.0),
                "beta": torch.empty(C, device=self.device, dtype=dtype).uniform_(-5.0, 5.0),
                "output": torch.empty((N, C), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # medium_size
        N, C = 64, 32
        tests.append(
            {
                "input": torch.empty((N, C), device=self.device, dtype=dtype).uniform_(-10.0, 10.0),
                "gamma": torch.empty(C, device=self.device, dtype=dtype).uniform_(0.5, 2.0),
                "beta": torch.empty(C, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                "output": torch.empty((N, C), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # single_feature
        N, C = 100, 1
        tests.append(
            {
                "input": torch.empty((N, C), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "gamma": torch.tensor([1.5], device=self.device, dtype=dtype),
                "beta": torch.tensor([0.5], device=self.device, dtype=dtype),
                "output": torch.empty((N, C), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # high_variance
        N, C = 10, 5
        input_data = torch.empty((N, C), device=self.device, dtype=dtype)
        for i in range(C):
            input_data[:, i] = torch.linspace(
                -100 + i * 10, 100 - i * 10, N, device=self.device, dtype=dtype
            )
        tests.append(
            {
                "input": input_data,
                "gamma": torch.ones(C, device=self.device, dtype=dtype),
                "beta": torch.zeros(C, device=self.device, dtype=dtype),
                "output": torch.empty((N, C), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N, C = 5000, 512
        return {
            "input": torch.empty((N, C), device=self.device, dtype=dtype).uniform_(-10.0, 10.0),
            "gamma": torch.empty(C, device=self.device, dtype=dtype).uniform_(0.5, 2.0),
            "beta": torch.empty(C, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
            "output": torch.empty((N, C), device=self.device, dtype=dtype),
            "N": N,
            "C": C,
            "eps": 1e-5,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_batch_normalization(...):
#     ...


def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    # TODO: launch your TileLang kernel and write the result into `output` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
