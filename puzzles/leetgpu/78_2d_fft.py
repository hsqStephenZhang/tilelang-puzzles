r"""
LeetGPU 78: 2D FFT
==================

Category: ["leetgpu"]
Difficulty: ["medium"]

Compute the 2D Discrete Fourier Transform (2D DFT) of a complex-valued signal stored on the GPU.
Given a 2D complex input signal of shape `M × N`, compute its 2D DFT spectrum
using the row-column decomposition: apply a 1D DFT along each row, then a 1D DFT along each
column of the result. All values are 32-bit floating point.

Implementation Requirements
---------------------------

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in `spectrum`

-
The input and output are stored as 1D arrays of interleaved real and imaginary parts in
row-major order: element `x[m, n]` has its real part at index
`2*(m*N + n)` and imaginary part at index `2*(m*N + n) + 1`

Example
-------

Input: `M` = 2, `N` = 2

Signal \(x[m, n]\) (real part):
\[
\begin{bmatrix}
1.0 & 0.0 \\
0.0 & 0.0
\end{bmatrix}
\]
Signal \(x[m, n]\) (imaginary part):
\[
\begin{bmatrix}
0.0 & 0.0 \\
0.0 & 0.0
\end{bmatrix}
\]
Output:

Spectrum \(X[k, l]\) (real part):
\[
\begin{bmatrix}
1.0 & 1.0 \\
1.0 & 1.0
\end{bmatrix}
\]
Spectrum \(X[k, l]\) (imaginary part):
\[
\begin{bmatrix}
0.0 & 0.0 \\
0.0 & 0.0
\end{bmatrix}
\]

Constraints
-----------

- 1 ≤ `M`, `N` ≤ 4096
- Signal values are 32-bit floating point (real and imaginary parts)
- Performance is measured with `M` = 2,048, `N` = 2,048

Run `python3 puzzles/leetgpu/78_2d_fft.py` to test your `solve`, or add `--ref` to run the
reference implementation through the same harness.
"""

import ctypes
from typing import Any, Dict, List

import tilelang  # noqa: F401
import tilelang.language as T  # noqa: F401
import torch
from common.leetgpu import ChallengeBase, run_challenge


class Challenge(ChallengeBase):
    name = "2D FFT"
    atol = 0.01
    rtol = 0.01
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, signal: torch.Tensor, spectrum: torch.Tensor, M: int, N: int):
        assert signal.shape == (M * N * 2,)
        assert spectrum.shape == (M * N * 2,)
        assert signal.dtype == torch.float32
        assert spectrum.dtype == torch.float32
        assert signal.device == spectrum.device

        sig_ri = signal.view(M, N, 2)
        sig_c = torch.complex(sig_ri[..., 0].contiguous(), sig_ri[..., 1].contiguous())
        spec_c = torch.fft.fft2(sig_c)
        spec_ri = torch.stack((spec_c.real, spec_c.imag), dim=-1).contiguous()
        spectrum.copy_(spec_ri.view(-1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "signal": (ctypes.POINTER(ctypes.c_float), "in"),
            "spectrum": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, N = 2, 2
        signal = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype
        )
        spectrum = torch.empty(M * N * 2, device=self.device, dtype=dtype)
        return {"signal": signal, "spectrum": spectrum, "M": M, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        cases = []

        def make_case(M, N, low=-1.0, high=1.0):
            signal = torch.empty(M * N * 2, device=self.device, dtype=dtype).uniform_(low, high)
            spectrum = torch.empty(M * N * 2, device=self.device, dtype=dtype)
            return {"signal": signal, "spectrum": spectrum, "M": M, "N": N}

        def make_zero_case(M, N):
            signal = torch.zeros(M * N * 2, device=self.device, dtype=dtype)
            spectrum = torch.empty(M * N * 2, device=self.device, dtype=dtype)
            return {"signal": signal, "spectrum": spectrum, "M": M, "N": N}

        def make_impulse_case(M, N):
            signal = torch.zeros(M * N * 2, device=self.device, dtype=dtype)
            signal[0] = 1.0
            spectrum = torch.empty(M * N * 2, device=self.device, dtype=dtype)
            return {"signal": signal, "spectrum": spectrum, "M": M, "N": N}

        # Edge cases: small sizes
        cases.append(make_impulse_case(1, 1))
        cases.append(make_zero_case(2, 2))
        cases.append(make_case(1, 4))

        # Power-of-2 sizes
        cases.append(make_case(16, 16))
        cases.append(make_case(32, 64))

        # Non-power-of-2 sizes
        cases.append(make_case(3, 5))
        cases.append(make_case(30, 30))

        # Mixed positive/negative values
        cases.append(make_case(100, 200, low=-5.0, high=5.0))

        # Realistic sizes
        cases.append(make_case(256, 256))
        cases.append(make_case(512, 512))

        return cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, N = 2048, 2048
        signal = torch.empty(M * N * 2, device=self.device, dtype=dtype).normal_(0.0, 1.0)
        spectrum = torch.empty(M * N * 2, device=self.device, dtype=dtype)
        return {"signal": signal, "spectrum": spectrum, "M": M, "N": N}


# TODO: Implement your TileLang kernel(s) here, e.g.
#
# @tilelang.jit
# def tl_2d_fft(...):
#     ...


def solve(
    signal: torch.Tensor,
    spectrum: torch.Tensor,
    M: int,
    N: int,
):
    # TODO: launch your TileLang kernel and write the result into `spectrum` in place.
    pass


if __name__ == "__main__":
    run_challenge(Challenge(), solve)
