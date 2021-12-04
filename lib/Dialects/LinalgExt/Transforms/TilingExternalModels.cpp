//===- TilingExternalModels.cpp - External models for TilingInterface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgExt/Passes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::linalg;

static Value getAsValue(OpBuilder &b, Location loc, OpFoldResult ofr) {
  if (auto v = ofr.dyn_cast<Value>())
    return v;
  return b.create<arith::ConstantIndexOp>(
      loc, ofr.get<Attribute>().cast<IntegerAttr>().getInt());
}
static SmallVector<Value> getAsValues(OpBuilder &b, Location loc,
                                      ArrayRef<OpFoldResult> ofrs) {
  SmallVector<Value> vals;
  vals.reserve(ofrs.size());
  for (auto ofr : ofrs)
    vals.push_back(getAsValue(b, loc, ofr));
  return vals;
}

namespace {

/// External model implementation of TilingInterface for LinalgOps. This is
/// templated on the actual Linalg named op for now since the registration of
/// the external model requires the original operation.
template <typename LinalgOpTy>
struct LinalgOpTilingInterface
    : public TilingInterface::ExternalModel<LinalgOpTilingInterface<LinalgOpTy>,
                                            LinalgOpTy> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    return linalgOp.getOutputOperands();
  }

  SmallVector<StringRef> getLoopIteratorTypes(Operation *op) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    SmallVector<StringRef> iteratorTypes;
    iteratorTypes.reserve(linalgOp.iterator_types().size());
    for (Attribute iteratorAttr : linalgOp.iterator_types()) {
      iteratorTypes.push_back(iteratorAttr.cast<StringAttr>().getValue());
    }
    return iteratorTypes;
  }

  SmallVector<Range> getLoopBounds(Operation *op, OpBuilder &b) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    return linalgOp.createLoopRanges(b, op->getLoc());
  }

  Operation *getTiledImplementation(Operation *op, OpBuilder &b,
                                    ValueRange dest,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    Location loc = op->getLoc();
    AffineMap shapeSizesToLoopsMap = linalgOp.getShapesToLoopsMap();
    auto allShapeSizes = linalgOp.createFlatListOfOperandDims(b, loc);
    if (!shapeSizesToLoopsMap)
      return nullptr;
    auto sizeBounds =
        applyMapToValues(b, loc, shapeSizesToLoopsMap, allShapeSizes);
    SmallVector<Value> valuesToTile = linalgOp.getInputOperands();
    valuesToTile.append(dest.begin(), dest.end());
    SmallVector<Value> tileOffsets = getAsValues(b, loc, offsets);
    SmallVector<Value> tileSizes = getAsValues(b, loc, sizes);
    SmallVector<Value, 4> tiledOperands = makeTiledShapes(
        b, loc, linalgOp, valuesToTile, tileOffsets, tileSizes, sizeBounds);

    SmallVector<Type, 4> resultTensorTypes;
    for (OpOperand *opOperand : linalgOp.getOutputTensorOperands()) {
      resultTensorTypes.push_back(
          tiledOperands[opOperand->getOperandNumber()].getType());
    }

    LinalgOp res = linalgOp.clone(b, loc, resultTensorTypes, tiledOperands);
    unsigned resultIdx = 0;
    SmallVector<Value> tensorResults;
    for (OpOperand *opOperand : linalgOp.getOutputTensorOperands()) {
      Value outputTensor = tiledOperands[opOperand->getOperandNumber()];
      if (auto sliceOp = outputTensor.getDefiningOp<tensor::ExtractSliceOp>()) {
        Value insertSlice = b.create<tensor::InsertSliceOp>(
            loc, res->getResult(resultIdx), sliceOp.source(),
            sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
            sliceOp.getMixedStrides());
        tensorResults.push_back(insertSlice);
      } else {
        tensorResults.push_back(res->getResult(resultIdx));
      }
      ++resultIdx;
    }
    return res;
  }
};
} // namespace

void mlir::linalg_ext::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addOpInterface<linalg::MatmulOp,
                          LinalgOpTilingInterface<linalg::MatmulOp>>();
}
