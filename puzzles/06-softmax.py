"""
Puzzle 06: Softmax
==============
Softmax is the first fundermental NN operator we learn in this tutorial.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

r"""
Softmax operator goes a little beyond the reduce sum. We also need to use serial loop to
accumulate the summation. And we need to perform an element-wise exp operation on each element
at the same time.

Note that softmax needs to be computed in numerically stable form as in Python. To achieve this,
we need to subtract the maximum value of each row from all elements in that row
before applying the exponential function.

HINT:
1. Use `T.fill` to set the initial value of the buffer. `T.clear` sets all elements to zero by
default, which may not be what you want.

3.We recommend not using `T.exp` but instead using `T.exp2`. You need the identity

.. math::
    \exp(x) = 2^{\log_2(e) x}

The constant log2_e is provided.

BONUS: Use "Online Softmax" algorithm to implement optimized softmax. This is also a core idea of
FlashAttention algorithm. Through this, we can implement softmax with only two passes / loops.

06-1: Softmax.

Inputs:
    A: Tensor([N, M], float32)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 4096
    M: int   # size of the tensor. 1 <= M <= 16384

Output:
    B: Tensor([N, M], float16)  # output tensor

Intermediates:
    MAX: float32  # max value of each row
    SUM: float32  # summation of each row

Definition:
    for i in range(N):
        SUM = 0
        MAX = -inf
        for j in range(M):
            MAX = max(A[i, j], MAX)
        for j in range(M):
            B[i, j] = exp(A[i, j] - MAX)
            SUM += B[i, j]
        for j in range(M):
            B[i, j] /= SUM
"""


def ref_softmax(A: torch.Tensor):
    assert len(A.shape) == 2
    assert A.dtype == torch.float32
    return torch.softmax(A, dim=1)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_softmax(A, BLOCK_N: int, BLOCK_M: int):
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)

    # naive impl: 一个 block 负责一行; 不使用 shared mem
    with T.Kernel(N, threads=256) as row:
        # 逻辑上，所有 threads 共享一个 BLOCK_M 的寄存器空间
        # 因为对 M 这个维度来说，我们都是先处理 (1, BLOCK_M) 个的局部元素，然后做全局的合并
        x = T.alloc_fragment((1, BLOCK_M), dtype)
        row_max = T.alloc_fragment((1,), dtype)
        row_sum = T.alloc_fragment((1,), dtype)

        T.fill(row_max, -T.infinity(dtype))

        for k in T.serial(T.ceildiv(M, BLOCK_M)):
            for j in T.Parallel(BLOCK_M):
                col = k * BLOCK_M + j
                x[0, j] = T.if_then_else(col < M, A[row, col], -T.infinity(dtype))

            T.reduce_max(x, row_max, dim=1, clear=False)

        T.clear(row_sum)

        for k in T.serial(T.ceildiv(M, BLOCK_M)):
            for j in T.Parallel(BLOCK_M):
                col = k * BLOCK_M + j
                x[0, j] = T.if_then_else(col < M, A[row, col], -T.infinity(dtype))
                x[0, j] = T.exp2((x[0, j] - row_max[0]) * log2_e)

            T.reduce_sum(x, row_sum, dim=1, clear=False)

        # 现在 `row_max` `row_sum` 都已经求出来, 且 `x` 已经完成使命了
        for k in T.serial(T.ceildiv(M, BLOCK_M)):
            for j in T.Parallel(BLOCK_M):
                col = k * BLOCK_M + j
                if col < M:
                    B[row, col] = T.exp2((A[row, col] - row_max[0]) * log2_e) / row_sum[0]

    return B


"""
将 max 和 sum 合并到一个 pass 中，从而减少对 A 的读取的负担，也叫 online 算法
其原理是，我们求出 max 的目的，其实是为了求出 sum(e^(val - max))
而每一组 [k * block_m, (k+1) * block_m)，实际上都有一个局部的最大值 max_local，
且都能获知前一个组的局部最大值 max_local_old
因此，我们在求每一组的 sum 的时候，实际都要依据前一组的来做修正，根据指数运算的性质，
旧的 sum 乘以 `e^(old_max - new_max)` 就可以实现这个效果 (new_max >= old_max, 因此该因子 <= 1)
S_new = S_old * e^(old_max - new_max) + sum(e^(x - new_max))
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_softmax_online(A, BLOCK_N: int, BLOCK_M: int):
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)

    with T.Kernel(N, threads=256) as row:
        # 逻辑上，所有 threads 共享一个 BLOCK_M 的寄存器空间
        # 因为对 M 这个维度来说，我们都是先处理 (1, BLOCK_M) 个的局部元素，然后做全局的合并
        x = T.alloc_fragment((1, BLOCK_M), dtype)
        row_max = T.alloc_fragment((1,), dtype)
        row_sum = T.alloc_fragment((1,), dtype)

        T.fill(row_max, -T.infinity(dtype))
        T.clear(row_sum)

        for k in T.serial(T.ceildiv(M, BLOCK_M)):
            for j in T.Parallel(BLOCK_M):
                col = k * BLOCK_M + j
                x[0, j] = T.if_then_else(col < M, A[row, col], -T.infinity(dtype))

            # 保存上一个 [(k-1) * block_m, k * block_m] 的最大值
            row_max_old = row_max[0]
            # 计算新的最大值 [k * block_m, (k + 1) * block_m]
            T.reduce_max(x, row_max, dim=1, clear=False)

            # 修正并保存到 row_sum 中
            row_sum_old_fixed = row_sum[0] * T.exp2((row_max_old - row_max[0]) * log2_e)

            # 计算新的 block_m 的和
            for j in T.Parallel(BLOCK_M):
                x[0, j] = T.exp2((x[0, j] - row_max[0]) * log2_e)

            T.reduce_sum(
                x, row_sum, dim=1, clear=True
            )  # clear 必须要设置为 True，清除上一轮计算的结果

            # 两者加到一起
            row_sum[0] = row_sum_old_fixed + row_sum[0]

        # 现在 `row_max` `row_sum` 都已经求出来, 且 `x` 已经完成使命了
        for k in T.serial(T.ceildiv(M, BLOCK_M)):
            for j in T.Parallel(BLOCK_M):
                col = k * BLOCK_M + j
                if col < M:
                    B[row, col] = T.exp2((A[row, col] - row_max[0]) * log2_e) / row_sum[0]

    return B


def run_softmax():
    print("\n=== Softmax ===\n")
    N = 4096
    M = 16384
    BLOCK_N = 16
    BLOCK_M = 256
    test_puzzle(
        tl_softmax_online,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )
    bench_puzzle(
        tl_softmax_online,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_softmax()
