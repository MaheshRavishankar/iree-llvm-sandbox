import sys, time
from collections.abc import Callable

import numpy as np

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.execution_engine import *
from mlir.runtime import *

from harness import *
from experts import *
from compilation import compile_and_callback, f32


def compile_and_test_linalg_matmul(M: int,
                                   N: int,
                                   K: int,
                                   ITERS: int,
                                   np_type: np.dtype,
                                   transform: Callable,
                                   dry_run: bool = True):
  A = np.random.rand(M, K).astype(np_type)
  B = np.random.rand(K, N).astype(np_type)
  C = np.random.rand(M, N).astype(np_type)
  C.fill(0.)

  # Arguments must be passed as pointers.
  A_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(A)))
  B_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(B)))
  C_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(C)))
  index_ptr_t = ctypes.c_longlong * 1

  def callback(module, execution_engine):

    def execute(m, n, k, iters):
      execution_engine.invoke('main', A_memref_ptr, B_memref_ptr, C_memref_ptr,
                              index_ptr_t(iters))

    if dry_run:
      # Dry-run.
      n_iters_dry_run = 1
      elapsed_s_per_iter, gflop_per_s_per_iter = timed_invoke(
          execute, n_iters_dry_run, M, N, K, n_iters_dry_run)
      print(f'dry_run in {elapsed_s_per_iter:.{4}}s per iter '
            f'sec ({gflop_per_s_per_iter:.{4}} GFlop/s) ')

    # Run for ITERS and report timing.
    elapsed_s_per_iter, gflop_per_s_per_iter = timed_invoke(
        execute, ITERS, M, N, K, ITERS)
    print(f'run in {elapsed_s_per_iter:.{4}}s per iter '
          f'sec ({gflop_per_s_per_iter:.{4}} GFlop/s) ')

    # Check results vs NP and print timings.
    success = 'SUCCESS' if np.allclose(C, np.dot(A, B)) else 'FAILURE'
    if success == 'SUCCESS':
      print(f'{success} ')
    else:
      delta = C - np.dot(A, B)
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      print(f'max_abs_delta: {max_abs_delta} -> {success} ')

  compile_and_callback(
      linalg.matmul, transform, callback, M=M, N=N, K=K, T1=f32, T2=f32, U=f32)


def test_numpy_matmul(M: int, N: int, K: int, ITERS, np_type):
  A = np.random.rand(M, K).astype(np_type)
  B = np.random.rand(K, N).astype(np_type)
  C = np.random.rand(M, N).astype(np_type)
  C.fill(0.)

  def execute(m, n, k, iters):
    for iters in range(iters):
      # TODO: True GEMM semantics ?
      C.fill(0.)
      np.dot(A, B, out=C)

  # Dry-run.
  n_iters_dry_run = 1
  elapsed_s_per_iter, gflop_per_s_per_iter = timed_invoke(
      execute, n_iters_dry_run, M, N, K, n_iters_dry_run)
  print(f'xxxxxxxxxx : numpy dry_run time on {1} threads '
        f'in {elapsed_s_per_iter:.{4}}s per iter '
        f'sec ({gflop_per_s_per_iter:.{4}} GFlop/s) ')

  # Run for ITERS and report timing.
  elapsed_s_per_iter, gflop_per_s_per_iter = timed_invoke(
      execute, ITERS, M, N, K, ITERS)
  print(f'xxxxxxxxxx : numpy time on {1} threads '
        f'in {elapsed_s_per_iter:.{4}}s per iter '
        f'sec ({gflop_per_s_per_iter:.{4}} GFlop/s) ')


def test_torch_matmul(M: int, N: int, K: int, ITERS: int, np_type,
                      num_threads: int):
  import torch
  torch.set_num_threads(num_threads)
  A = torch.rand(M, K)
  B = torch.rand(K, N)
  C = torch.rand(M, N)
  C.fill_(0.)

  def execute(m, n, k, iters):
    for iters in range(iters):
      # TODO: True GEMM semantics ?
      C.fill_(0.)
      torch.mm(A, B, out=C)

  # Dry-run.
  n_iters_dry_run = 1
  elapsed_s_per_iter, gflop_per_s_per_iter = timed_invoke(
      execute, n_iters_dry_run, M, N, K, n_iters_dry_run)
  print(f'xxxxxxxxxx : torch dry_run time on {torch.get_num_threads()} threads '
        f'in {elapsed_s_per_iter:.{4}}s per iter '
        f'sec ({gflop_per_s_per_iter:.{4}} GFlop/s) ')

  # Run for ITERS and report timing.
  elapsed_s_per_iter, gflop_per_s_per_iter = timed_invoke(
      execute, ITERS, M, N, K, ITERS)
  print(f'xxxxxxxxxx : torch time on {torch.get_num_threads()} threads '
        f'in {elapsed_s_per_iter:.{4}}s per iter '
        f'sec ({gflop_per_s_per_iter:.{4}} GFlop/s) ')
