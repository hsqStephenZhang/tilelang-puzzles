"""
Minimal local runner for challenges copied from LeetGPU (puzzles/leetgpu).

The base classes mirror `challenges/core/challenge_base.py` in leetgpu-challenges so the copied
`Challenge` classes work unchanged. `run_challenge` calls `solve(**case)` on each test case and
compares every "out"/"inout" tensor against `reference_impl`.
"""

import sys
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch

from common.utils import time_cuda


class RandTensor:
    """Uniform random input in [low, high)."""

    def __init__(self, shape, low=0.0, high=1.0, dtype="float32"):
        self.shape = tuple(shape)
        self.low = low
        self.high = high
        self.dtype = dtype


class RandnTensor:
    """Normal (Gaussian) random input."""

    def __init__(self, shape, mean=0.0, std=1.0, dtype="float32"):
        self.shape = tuple(shape)
        self.mean = mean
        self.std = std
        self.dtype = dtype


class RandIntTensor:
    """Uniform integer random input in [low, high)."""

    def __init__(self, shape, low, high, dtype="int32"):
        self.shape = tuple(shape)
        self.low = low
        self.high = high
        self.dtype = dtype


class FullTensor:
    """Constant-filled input (covers zeros / ones / full)."""

    def __init__(self, shape, value=0.0, dtype="float32"):
        self.shape = tuple(shape)
        self.value = value
        self.dtype = dtype


class OutTensor:
    """Output buffer (by shape), materialized as an empty tensor."""

    def __init__(self, shape, dtype="float32"):
        self.shape = tuple(shape)
        self.dtype = dtype


class ChallengeBase(ABC):
    name: str
    atol: float
    rtol: float
    num_gpus: int
    access_tier: str

    def __init__(self, device: str = "cuda"):
        self.device = device

    @abstractmethod
    def reference_impl(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_solve_signature(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def generate_example_test(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def generate_functional_test(self) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def generate_performance_test(self) -> dict[str, Any]:
        pass


def _materialize(value, device):
    """Turn the lazy Rand*/Full/Out tensor specs used by performance tests into tensors."""

    if not hasattr(value, "shape") or isinstance(value, torch.Tensor):
        return value
    dtype = getattr(torch, value.dtype)
    if isinstance(value, RandTensor):
        return torch.empty(value.shape, dtype=dtype, device=device).uniform_(value.low, value.high)
    if isinstance(value, RandnTensor):
        return torch.randn(value.shape, device=device).mul_(value.std).add_(value.mean).to(dtype)
    if isinstance(value, RandIntTensor):
        return torch.randint(value.low, value.high, value.shape, dtype=dtype, device=device)
    if isinstance(value, FullTensor):
        return torch.full(value.shape, value.value, dtype=dtype, device=device)
    if isinstance(value, OutTensor):
        return torch.empty(value.shape, dtype=dtype, device=device)
    return value


def _prepare(case: dict, device) -> dict:
    return {k: _materialize(v, device) for k, v in case.items()}


def _clone(case: dict) -> dict:
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in case.items()}


def _check_case(challenge: ChallengeBase, solve: Callable, case: dict, label: str) -> bool:
    outputs = [k for k, (_, d) in challenge.get_solve_signature().items() if d in ("out", "inout")]
    expected, actual = _clone(case), _clone(case)
    challenge.reference_impl(**expected)
    solve(**actual)
    torch.cuda.synchronize()

    for name in outputs:
        exp, act = expected[name], actual[name]
        if exp.dtype.is_floating_point:
            ok = torch.allclose(act, exp, atol=challenge.atol, rtol=challenge.rtol, equal_nan=True)
        else:
            ok = torch.equal(act, exp)
        if not ok:
            diff = (act.double() - exp.double()).abs()
            print(f"❌ {label}: `{name}` mismatch, max diff {diff.max().item():.4g}")
            print("Yours:", act.dtype, tuple(act.shape), "\n", act)
            print("Spec:", exp.dtype, tuple(exp.shape), "\n", exp)
            return False
    print(f"✅ {label}")
    return True


def run_challenge(challenge: ChallengeBase, solve: Callable):
    """Run the example, functional and performance tests of a challenge against `solve`.

    Pass `--ref` on the command line to use `reference_impl` as the solution (checks the harness).
    """

    if "--ref" in sys.argv:
        solve = challenge.reference_impl
    torch.manual_seed(0)
    print(f"\n=== {challenge.name} ===\n")

    cases = [("example", challenge.generate_example_test())]
    cases += [(f"functional #{i}", c) for i, c in enumerate(challenge.generate_functional_test())]
    cases = [(label, _prepare(case, challenge.device)) for label, case in cases]
    passed = sum(_check_case(challenge, solve, case, label) for label, case in cases)
    print(f"\n{passed}/{len(cases)} tests passed")
    if passed != len(cases):
        return

    perf = challenge.generate_performance_test()
    if isinstance(perf, list):
        perf = perf[0]
    perf = _prepare(perf, challenge.device)
    ref_case, tl_case = _clone(perf), _clone(perf)
    torch_time = time_cuda(lambda: challenge.reference_impl(**ref_case), warmups=3, repeats=20)
    tl_time = time_cuda(lambda: solve(**tl_case), warmups=3, repeats=20)
    print(f"Torch time: {torch_time:.3f} ms")
    print(f"Tilelang time: {tl_time:.3f} ms")
