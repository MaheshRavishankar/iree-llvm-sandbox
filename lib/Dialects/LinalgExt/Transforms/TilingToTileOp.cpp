//===- TilingToTileOp.cpp - Tiling using to TileOp TilingInterface --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Dialects/LinalgExt/LinalgExtOps.h"
#include "Dialects/LinalgExt/PassDetail.h"
#include "Dialects/LinalgExt/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>

using namespace mlir;
using namespace mlir::linalg_ext;

SmallVector<Value> tileToTileOp(PatternRewriter &rewriter, TilingInterface op,
                                Value tileSize) {
  Location loc = op->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  // TODO: Handle the case where the `loopRanges` are empty.
  SmallVector<Range> loopRanges = op.getLoopBounds(rewriter);
  assert(loopRanges.size() >= 1 &&
         "expected at least a single loop in operation");
  auto tileOp = rewriter.create<linalg_ext::TileOp>(
      loc, tileSize, op.getDestinationOperands(rewriter),
      [&](OpBuilder &b, Location loc, Value offset, Value size,
          ValueRange outSlices) {
        // TODO: support `getTiledImplementation` with >1 produced tiled ops.
        SmallVector<OpFoldResult> tiledOffsets = {offset}, tiledSizes = {size};
        for (unsigned i = 1; i < loopRanges.size(); ++i) {
          tiledOffsets.push_back(loopRanges[i].offset);
          tiledSizes.push_back(loopRanges[i].size);
        }
        Operation *tiledOp =
            op.getTiledImplementation(b, outSlices, tiledOffsets, tiledSizes);
        b.create<linalg_ext::TileYieldOp>(loc, tiledOp->getResults());
      });
  return tileOp->getResults();
}

namespace {

struct OpTilingPattern : public OpInterfaceRewritePattern<TilingInterface> {
  OpTilingPattern(MLIRContext *context, linalg::LinalgTilingOptions opt)
      : OpInterfaceRewritePattern<TilingInterface>(context), options(opt) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    // Poor man's single pattern application to avoid infinite pattern
    // application wihtout having to hack with attributes and filters.
    if (successfulApplicationCount <= 0)
      return failure();

    /// Currently only handle single result operations.
    if (op->getNumResults() != 1)
      return failure();

    /// Currently only handle operations with all parallel iterator types.
    auto loopIteratorTypes = op.getLoopIteratorTypes();
    if (loopIteratorTypes.empty()) {
      // Scalar operation, nothing to do, so just return.
      return failure();
    }
    if (llvm::any_of(op.getLoopIteratorTypes(), [](StringRef iteratorType) {
          return iteratorType != getParallelIteratorTypeName();
        })) {
      return failure();
    }

    // Get rank and tile sizes.
    SmallVector<Value> tileSizes =
        options.tileSizeComputationFunction(rewriter, op);
    assert(tileSizes.size() == 1 && "expected a single tile size");

    rewriter.replaceOp(op, tileToTileOp(rewriter, op, tileSizes.front()));

    // Poor man's single pattern application to avoid infinite pattern
    // application wihtout having to hack with attributes and filters.
    --successfulApplicationCount;

    return success();
  }

private:
  linalg::LinalgTilingOptions options;
  linalg::LinalgTransformationFilter filter;
  // Poor man's single pattern application to avoid infinite pattern
  // application wihtout having to hack with attributes and filters.
  mutable int64_t successfulApplicationCount = 1;
};

/// Pass to test the tiling tranforamtion.
struct LinalgExtTilingToTileOp
    : public LinalgExtTilingToTileOpBase<LinalgExtTilingToTileOp> {
  LinalgExtTilingToTileOp() = default;
  LinalgExtTilingToTileOp(int64_t tileSize) { this->tileSize = tileSize; }
  void runOnOperation() override;
};
} // namespace

void LinalgExtTilingToTileOp::runOnOperation() {
  assert(tileSize > 0 && "expected a nonnegative tile size");

  FuncOp funcOp = getOperation();

  RewritePatternSet patterns(&getContext());
  auto options =
      linalg::LinalgTilingOptions().setTileSizes(ArrayRef<int64_t>{tileSize});
  patterns.insert<OpTilingPattern>(&getContext(), options);

  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::linalg_ext::createLinalgExtTilingToTileOpPass(int64_t tileSize) {
  return std::make_unique<LinalgExtTilingToTileOp>(tileSize);
}
