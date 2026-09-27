r"""
LeetGPU 25: Categorical Cross Entropy Loss
==========================================

Category: ["leetgpu"]
Difficulty: ["medium"]

Implement a GPU program to calculate the categorical cross-entropy loss for a batch of predictions.
Given a matrix of predicted logits \(Z\) of size \(N \times C\) and a vector of true class labels `true_labels` of size \(N\), compute the average cross-entropy loss over the batch.
The loss for a single sample \(j\) with logits \(z_j = [z_{j1}, \ldots, z_{jC}]\) and true label \(y_j\) is calculated using the numerically stable formula:
\[ \text{Loss}_j = \log\left(\sum_{k=1}^{C} e^{z_{jk}}\right) - z_{j, y_j} \]
The final output stored in the `loss` variable should be the average loss over the \(N\) samples:
\[ L = \frac{1}{N} \sum_{j=1}^{N} \text{Loss}_j \]
The input parameters are `logits`, `true_labels`, `N` (number of samples), and `C` (number of classes). The result should be stored in `loss` (a pointer to a single float).

Implementation Requirements
---------------------------

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result (average loss) must be stored in `loss`

Example 1:
----------

Input:  N = 2, C = 3
logits = [[1.0, 2.0, 0.5], [0.1, 3.0, 1.5]]
true_labels = [1, 1]
Output: loss = [0.3548926]

Example 2:
----------

Input:  N = 3, C = 4
logits = [[-0.5, 1.5, 0.0, 1.0], [2.0, -1.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]]
true_labels = [3, 0, 1]
Output: loss = [0.98820376]

Constraints
-----------

- 1 ≤ `N` ≤ 10,000
- 2 ≤ `C` ≤ 1,000
- -10.0 ≤ `logits[i, j]` ≤ 10.0
- 0 ≤ `true_labels[i]` ≤ `C`
- Performance is measured with `N` = 10,000

Run `python3 puzzles/leetgpu/25_categorical_cross_entropy_loss.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, OutTensor, RandIntTensor, RandTensor, run_challenge


class Challenge(ChallengeBase):
    name = "Categorical Cross Entropy Loss"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self, logits: torch.Tensor, true_labels: torch.Tensor, loss: torch.Tensor, N: int, C: int
    ):
        assert logits.dtype == torch.float32
        assert true_labels.dtype == torch.int32
        assert loss.dtype == torch.float32
        assert logits.shape == (N, C)
        assert true_labels.shape == (N,)
        assert loss.shape == (1,)
        assert N > 0 and C > 0
        total_loss = 0.0
        for i in range(N):
            log_probs = logits[i]
            true_label = true_labels[i].item()
            assert 0 <= true_label < C
            max_logit = torch.max(log_probs)
            log_sum_exp = max_logit + torch.log(torch.sum(torch.exp(log_probs - max_logit)))
            loss_i = log_sum_exp - log_probs[true_label]
            total_loss += loss_i.item()
        loss[0] = total_loss / N

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "true_labels": (ctypes.POINTER(ctypes.c_int), "in"),
            "loss": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype_logits = torch.float32
        dtype_labels = torch.int32
        logits = torch.tensor(
            [[1.0, 2.0, 0.5], [0.1, 3.0, 1.5]], device=self.device, dtype=dtype_logits
        )
        true_labels = torch.tensor([1, 1], device=self.device, dtype=dtype_labels)
        loss = torch.zeros(1, device=self.device, dtype=dtype_logits)
        return {
            "logits": logits,
            "true_labels": true_labels,
            "loss": loss,
            "N": 2,
            "C": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype_logits = torch.float32
        dtype_labels = torch.int32
        tests = []
        # basic_example
        tests.append(
            {
                "logits": torch.tensor(
                    [[1.0, 2.0, 0.5], [0.1, 3.0, 1.5]], device=self.device, dtype=dtype_logits
                ),
                "true_labels": torch.tensor([1, 1], device=self.device, dtype=dtype_labels),
                "loss": torch.zeros(1, device=self.device, dtype=dtype_logits),
                "N": 2,
                "C": 3,
            }
        )
        # example_2
        tests.append(
            {
                "logits": torch.tensor(
                    [[-0.5, 1.5, 0.0, 1.0], [2.0, -1.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]],
                    device=self.device,
                    dtype=dtype_logits,
                ),
                "true_labels": torch.tensor([3, 0, 1], device=self.device, dtype=dtype_labels),
                "loss": torch.zeros(1, device=self.device, dtype=dtype_logits),
                "N": 3,
                "C": 4,
            }
        )
        # single_sample
        tests.append(
            {
                "logits": torch.tensor(
                    [[0.1, 0.2, 0.3, 0.4, 0.5]], device=self.device, dtype=dtype_logits
                ),
                "true_labels": torch.tensor([4], device=self.device, dtype=dtype_labels),
                "loss": torch.zeros(1, device=self.device, dtype=dtype_logits),
                "N": 1,
                "C": 5,
            }
        )
        # uniform_logits_correct_label
        tests.append(
            {
                "logits": torch.tensor(
                    [[1.0] * 5, [1.0] * 5], device=self.device, dtype=dtype_logits
                ),
                "true_labels": torch.tensor([0, 0], device=self.device, dtype=dtype_labels),
                "loss": torch.zeros(1, device=self.device, dtype=dtype_logits),
                "N": 2,
                "C": 5,
            }
        )
        # high_confidence_correct
        tests.append(
            {
                "logits": torch.tensor(
                    [[-5.0, -5.0, 10.0, -5.0], [10.0, -5.0, -5.0, -5.0]],
                    device=self.device,
                    dtype=dtype_logits,
                ),
                "true_labels": torch.tensor([2, 0], device=self.device, dtype=dtype_labels),
                "loss": torch.zeros(1, device=self.device, dtype=dtype_logits),
                "N": 2,
                "C": 4,
            }
        )
        # high_confidence_incorrect
        tests.append(
            {
                "logits": torch.tensor(
                    [[10.0, -5.0, -5.0], [-5.0, 10.0, -5.0]], device=self.device, dtype=dtype_logits
                ),
                "true_labels": torch.tensor([1, 2], device=self.device, dtype=dtype_labels),
                "loss": torch.zeros(1, device=self.device, dtype=dtype_logits),
                "N": 2,
                "C": 3,
            }
        )
        # larger_batch_random
        tests.append(
            {
                "logits": torch.empty(100, 5, device=self.device, dtype=dtype_logits).uniform_(
                    -5.0, 5.0
                ),
                "true_labels": torch.randint(0, 5, (100,), device=self.device, dtype=dtype_labels),
                "loss": torch.zeros(1, device=self.device, dtype=dtype_logits),
                "N": 100,
                "C": 5,
            }
        )
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        logits = RandTensor((10000, 1000), -10.0, 10.0)
        true_labels = RandIntTensor((10000,), 0, 1000, dtype="int32")
        loss = OutTensor((1,))
        return {
            "logits": logits,
            "true_labels": true_labels,
            "loss": loss,
            "N": 10000,
            "C": 1000,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_categorical_cross_entropy_loss(...):
#     ...


def solve(
    logits: torch.Tensor,
    true_labels: torch.Tensor,
    loss: torch.Tensor,
    N: int,
    C: int,
):
    # TODO: launch your TileLang kernel and write the result into `loss` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
