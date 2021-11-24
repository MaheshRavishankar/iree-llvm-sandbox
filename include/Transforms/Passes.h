//===- Passes.h - Linalg pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_PASSES_H
#define IREE_LLVM_SANDBOX_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates a pass to drive bufferization.
std::unique_ptr<OperationPass<ModuleOp>> createLinalgBufferizationDriverPass();

/// Creates a pass to drive tile + fuse transformations.
std::unique_ptr<OperationPass<FuncOp>> createLinalgFusePass();

/// Creates a pass to driver fuse output into reduction transformations.
std::unique_ptr<OperationPass<FuncOp>>
createLinalgFuseOutputIntoReductionPass();

/// Creates a pass to drive transformations on Linalg on tensors.
std::unique_ptr<OperationPass<FuncOp>> createLinalgTensorCodegenDriverPass();

/// Creates a pass to driver the lowering of vector operations.
std::unique_ptr<OperationPass<FuncOp>> createLinalgVectorLoweringPass();

/// Creates a pass to driver lowering to LLVM.
std::unique_ptr<OperationPass<ModuleOp>> createLLVMLoweringPass();

/// Experimental pass for vector distribution.
std::unique_ptr<OperationPass<FuncOp>> createPropagateVectorDistribution();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"

}  // namespace mlir

#endif  // IREE_LLVM_SANDBOX_PASSES_H
