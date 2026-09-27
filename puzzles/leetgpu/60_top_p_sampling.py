r"""
LeetGPU 60: Top-p Sampling
==========================

Category: ["leetgpu"]
Difficulty: ["medium"]

Write a GPU program that implements top-p (nucleus) sampling for LLM inference.

Top-p sampling is a text generation technique where you sample from the smallest set of tokens whose cumulative probability exceeds threshold p.
This balances randomness and quality better than pure top-k or greedy sampling.

Given logits (unnormalized scores) from a language model:

- Convert logits to probabilities using softmax
- Sort tokens by probability (descending)
- Find the smallest set where cumulative probability ≥ p (the "nucleus")
- Renormalize the nucleus probabilities to sum to 1
- Sample a token from the nucleus using the provided random seed

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- Ensure numerical stability when computing softmax

Example 1:
----------

Input:
logits = [1.0, 2.0, 3.0, 0.5]
p = 0.9
seed = 42

Output:
sampled_token = 2 or 1
(tokens with highest probabilities, sampled randomly)

Example 2:
----------

Input:
logits = [10.0, 1.0, 1.0]
p = 0.5
seed = 123

Output:
sampled_token = 0
(single token dominates the probability mass)

Constraints
-----------

- 3 ≤ `vocab_size` ≤ 50,000
- -100.0 ≤ `logits[i]` ≤ 100.0
- 0.0 < `p` ≤ 1.0
- 0 ≤ `sampled_token` < vocab_size
- Performance is measured with `vocab_size` = 50,000

Run `python3 puzzles/leetgpu/60_top_p_sampling.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "Top-p Sampling"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        logits: torch.Tensor,
        p: torch.Tensor,
        seed: torch.Tensor,
        sampled_token: torch.Tensor,
        vocab_size: int,
    ):
        assert logits.shape == (vocab_size,)
        assert p.shape == (1,)
        assert seed.shape == (1,)
        assert sampled_token.shape == (1,)
        assert logits.dtype == torch.float32
        assert p.dtype == torch.float32

        p_value = p.item()
        seed_value = seed.item()

        max_logit = torch.max(logits)
        exp_logits = torch.exp(logits - max_logit)
        probs = exp_logits / torch.sum(exp_logits)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)

        cutoff_idx = torch.searchsorted(cumsum, p_value, right=False).item()
        cutoff_idx = min(cutoff_idx + 1, vocab_size)

        nucleus_probs = sorted_probs[:cutoff_idx]
        nucleus_indices = sorted_indices[:cutoff_idx]

        nucleus_probs = nucleus_probs / torch.sum(nucleus_probs)

        torch.manual_seed(seed_value)
        sampled_idx = torch.multinomial(nucleus_probs, 1).item()
        sampled_token[0] = nucleus_indices[sampled_idx]

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "p": (ctypes.POINTER(ctypes.c_float), "in"),
            "seed": (ctypes.POINTER(ctypes.c_int32), "in"),
            "sampled_token": (ctypes.POINTER(ctypes.c_int32), "out"),
            "vocab_size": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        logits = torch.tensor([1.0, 2.0, 3.0, 0.5], device=self.device, dtype=torch.float32)
        p = torch.tensor([0.9], device=self.device, dtype=torch.float32)
        seed = torch.tensor([42], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)

        return {
            "logits": logits,
            "p": p,
            "seed": seed,
            "sampled_token": sampled_token,
            "vocab_size": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        logits = torch.tensor([1.0, 2.0, 3.0], device=self.device, dtype=torch.float32)
        p = torch.tensor([0.95], device=self.device, dtype=torch.float32)
        seed = torch.tensor([123], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 3,
            }
        )

        logits = torch.randn(10, device=self.device, dtype=torch.float32)
        p = torch.tensor([0.9], device=self.device, dtype=torch.float32)
        seed = torch.tensor([456], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 10,
            }
        )

        logits = torch.randn(100, device=self.device, dtype=torch.float32) * 5.0
        p = torch.tensor([0.85], device=self.device, dtype=torch.float32)
        seed = torch.tensor([789], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 100,
            }
        )

        logits = torch.zeros(50, device=self.device, dtype=torch.float32)
        logits[0] = 10.0
        p = torch.tensor([0.5], device=self.device, dtype=torch.float32)
        seed = torch.tensor([111], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 50,
            }
        )

        logits = torch.randn(500, device=self.device, dtype=torch.float32) * 3.0
        p = torch.tensor([0.92], device=self.device, dtype=torch.float32)
        seed = torch.tensor([222], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 500,
            }
        )

        logits = torch.linspace(-5, 5, 200, device=self.device, dtype=torch.float32)
        p = torch.tensor([0.8], device=self.device, dtype=torch.float32)
        seed = torch.tensor([333], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 200,
            }
        )

        logits = torch.randn(1000, device=self.device, dtype=torch.float32) * 2.0
        p = torch.tensor([0.95], device=self.device, dtype=torch.float32)
        seed = torch.tensor([444], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 1000,
            }
        )

        logits = torch.randn(5000, device=self.device, dtype=torch.float32)
        p = torch.tensor([0.9], device=self.device, dtype=torch.float32)
        seed = torch.tensor([555], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 5000,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        vocab_size = 50000
        logits = torch.randn(vocab_size, device=self.device, dtype=torch.float32) * 3.0
        p = torch.tensor([0.9], device=self.device, dtype=torch.float32)
        seed = torch.tensor([999], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)

        return {
            "logits": logits,
            "p": p,
            "seed": seed,
            "sampled_token": sampled_token,
            "vocab_size": vocab_size,
        }


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_top_p_sampling(...):
#     ...


def solve(
    logits: torch.Tensor,
    p: torch.Tensor,
    seed: torch.Tensor,
    sampled_token: torch.Tensor,
    vocab_size: int,
):
    # TODO: launch your TileLang kernel and write the result into `sampled_token` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
