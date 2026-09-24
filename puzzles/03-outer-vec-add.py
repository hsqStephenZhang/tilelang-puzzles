"""
Puzzle 03: Outer Vector Add
==============
In this puzzle we will enter the 2D world!

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Consider an outer vector addition operation. The result is a matrix where
each element (i, j) is the sum of A[i] and B[j].

The main difference from the previous puzzle is that C is now a 2D tensor and
we have two different iterators in buffers A and B. So the dataflow is also
a little different.

But remember that any N dimensional tensor can be viewed as a 1D tensor in memory.
So we just need to handle the indexing properly.

03-1: Outer vector addition.

Inputs:
    A: Tensor([N,], float16)  # input tensor
    B: Tensor([M,], float16)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 8192
    M: int   # size of the tensor. 1 <= M <= 8192

Output:
    C: [N, M]  # output tensor

Definition:
    for i in range(N):
        for j in range(M):
            C[i, j] = A[i] + B[j]
"""


def ref_outer_add(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.dtype == B.dtype == torch.float16
    return torch.add(input=A[:, None], other=B[None, :])


"""
两个关键因素 (机器: GTX 1650 SUPER, 20 SM, Turing)

1. SM 利用率. 每个 SM 有最多可驻留的线程数, Turing 是 1024 (Ampere 以后是 2048), 即 4 个
   256 线程的 block, 全卡同时能跑 20 * 4 = 80 个 block. grid 至少要有 2~3 波, 也就是 160~240
   个 block 以上; 多几倍无妨, 几千个 block 也没有额外开销.
   本例 BLOCK_N=BLOCK_M=1024 时 grid 只有 8*4=32 个 block (0.4 波), ncu 显示 SM 活跃周期 68%,
   活跃 warp 36%, DRAM 带宽 62%, 耗时 0.69 ms; BLOCK_N 降到 128 (grid 256) 后 DRAM 92%,
   耗时 0.39 ms, 超过 torch 的 0.47 ms.

2. tile 形状. 每 block 线程数 (64~1024) 对性能几乎无影响, TileLang 都会生成 128-bit 访存.
   影响性能的是 tile 的最内维 BLOCK_M（C 的最内层）: 它决定一个 warp 一次写入的连续字节数,
   BLOCK_M=64 (128 B/行) 比 BLOCK_M>=1024 慢约 25%. BLOCK_N*BLOCK_M 只需大到能分摊
   shared memory 搬运和 __syncthreads 的固定开销, 本例 8K 元素/block 已足够.
"""


@tilelang.jit
def tl_outer_add(A, B, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float16
    A: T.Tensor((N,), dtype)
    B: T.Tensor((M,), dtype)
    C = T.empty((N, M), dtype)

    # TODO: Implement this function
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=256) as (bx, by):
        A_shared = T.alloc_shared((BLOCK_N,), dtype)
        B_shared = T.alloc_shared((BLOCK_M,), dtype)

        baseX = bx * BLOCK_N
        baseY = by * BLOCK_M

        T.copy(A[baseX : baseX + BLOCK_N], A_shared, coalesced_width=4)
        T.copy(B[baseY : baseY + BLOCK_M], B_shared, coalesced_width=4)

        for i in T.Parallel(BLOCK_N):
            for j in T.Parallel(BLOCK_M):
                C[baseX + i, baseY + j] = A_shared[i] + B_shared[j]

    return C


def run_outer_add():
    print("\n=== Outer Vector Add ===\n")
    N = 8192
    M = 4096
    BLOCK_N = 1024
    BLOCK_M = 1024
    test_puzzle(
        tl_outer_add,
        ref_outer_add,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )

    bench_puzzle(
        tl_outer_add,
        ref_outer_add,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        bench_torch=True,
        bench_name="Normal BLOCK_N"
    )

def run_outer_add_small_blockn():
    print("\n=== Outer Vector Add With Small BLOCK_N ===\n")
    N = 8192
    M = 4096
    BLOCK_N = 128
    BLOCK_M = 1024
    test_puzzle(
        tl_outer_add,
        ref_outer_add,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )

    bench_puzzle(
        tl_outer_add,
        ref_outer_add,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        bench_torch=True,
        bench_name="Small BLOCK_N"
    )

if __name__ == "__main__":
    run_outer_add()
    run_outer_add_small_blockn()
