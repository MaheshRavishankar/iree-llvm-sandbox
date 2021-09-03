//===- LinalgTensorCodegenStrategyPass.cpp - Test Linalg codegen strategy--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing the Linalg codegen strategy.
//
//===----------------------------------------------------------------------===//

// TODO: avoid copy-pasta but I can't seem to be able to inherit from a pass.
// Will get better once upstreamed to core and it replaces the existing codegen
// strategy.

#include "Transforms.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct LinalgTensorCodegenStrategyPass
    : public PassWrapper<LinalgTensorCodegenStrategyPass, FunctionPass> {
  LinalgTensorCodegenStrategyPass() = default;
  LinalgTensorCodegenStrategyPass(const LinalgTensorCodegenStrategyPass &pass) {
  }

  StringRef getArgument() const final {
    return "linalg-tensor-codegen-strategy";
  }
  StringRef getDescription() const final {
    return "Linalg Tensor Codegen Strategy.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    async::AsyncDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    StandardOpsDialect,
                    tensor::TensorDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  template <typename LinalgNamedOp>
  void applyStrategyToNamedLinalgOp();

  void runOnFunction() override;

  ListOption<int64_t> tileSizes{*this, "tile-sizes",
                                llvm::cl::MiscFlags::CommaSeparated,
                                llvm::cl::desc("Specifies the tile sizes.")};
  Option<bool> vectorize{
      *this, "vectorize",
      llvm::cl::desc("Rewrite the linalg op as a vector operation."),
      llvm::cl::init(false)};
  Option<std::string> splitVectorTransfersTo{
      *this, "split-transfers",
      llvm::cl::desc(
          "Split vector transfers between slow (masked) and fast "
          "(unmasked) variants. Possible options are:\n"
          "\tnone: keep unsplit vector.transfer and pay the full price\n"
          "\tlinalg-copy: use linalg.fill + linalg.copy for the slow path\n"
          "\tvector-transfers: use extra small unmasked vector.transfer for"
          " the slow path\n"),
      llvm::cl::init("none")};
  Option<std::string> vectorizeContractionTo{
      *this, "vectorize-contraction-to",
      llvm::cl::desc("the type of vector op to use for linalg contractions"),
      llvm::cl::init("outerproduct")};
  Option<bool> unrollVectorTransfers{
      *this, "unroll-vector-transfers",
      llvm::cl::desc("Enable full unrolling of vector.transfer operations"),
      llvm::cl::init(false)};
  Option<bool> licm{*this, "licm", llvm::cl::desc("Enable LICM."),
                    llvm::cl::init(true)};
  Option<bool> hoistRedundantVectorTransfers{
      *this, "hoist-redundant-vector-transfers",
      llvm::cl::desc("Enable HoistRedundantVectorTransfers"),
      llvm::cl::init(true)};
  Option<bool> vectorTransferPartialRewrite{
      *this, "vector-transfer-partial-rewrite",
      llvm::cl::desc("Enable rewriting of vector.transfer operations into "
                     "full/partial read/writes."),
      llvm::cl::init(true)};
  Option<bool> vectorContractLowering{
      *this, "vector-contract-lowering",
      llvm::cl::desc("Enable lowering of vector contractions."),
      llvm::cl::init(true)};
  Option<bool> vectorToSCFConversion{
      *this, "vector-to-scf-conversion",
      llvm::cl::desc("Enable vector to scf conversions."),
      llvm::cl::init(true)};
  Option<std::string> anchorOpName{
      *this, "anchor-op",
      llvm::cl::desc(
          "Which single linalg op is the anchor for the codegen strategy to "
          "latch on:\n"
          "\tlinalg.matmul: anchor on linalg.matmul\n"
          "\tlinalg.matmul_column_major: anchor on linalg.matmul_column_major\n"
          "\tlinalg.copy: anchor on linalg.copy\n"
          "\tlinalg.fill: anchor on linalg.fill\n"),
      llvm::cl::init("")};
  Option<std::string> anchorFuncOpName{
      *this, "anchor-func",
      llvm::cl::desc(
          "Which single func op is the anchor for the codegen strategy to "
          "latch on."),
      llvm::cl::init("")};

  Option<bool> distribute{
      *this, "distribute",
      llvm::cl::desc("Distribute the linalg op into a TiledGeneric."),
      llvm::cl::init(false)};
  ListOption<int64_t> distributeTileSizes{
      *this, "distribute-tile-sizes", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Specifies the tile sizes.")};
  ListOption<std::string> distributionTypes{
      *this, "distribution-types", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc(
          "Specifies the markers how to distribute each loop dimension.")};
  Option<bool> pad{*this, "pad", llvm::cl::desc("Use padding during tiling."),
                   llvm::cl::init(false)};
  Option<int> hoistPadding{
      *this, "hoist-padding",
      llvm::cl::desc("Hoist padding by the number of specified loops."),
      llvm::cl::init(0)};
  Option<bool> vectorizePadding{
      *this, "vectorize-padding",
      llvm::cl::desc("Rewrite linalg.pad_tensor in vector form."),
      llvm::cl::init(false)};
  Option<bool> fuse{*this, "fuse", llvm::cl::desc("Fuse."),
                    llvm::cl::init(false)};
  Option<bool> fusePadding{*this, "fuse-padding",
                           llvm::cl::desc("Use padding during fusion."),
                           llvm::cl::init(false)};
  Option<bool> tiledLoopToGPU{
      *this, "convert-to-gpu",
      llvm::cl::desc("Wrap the top level tiled.loop op in a gpu.launch region"),
      llvm::cl::init(false)};
  ListOption<int64_t> numberGPUWorkgroups{
      *this, "num-gpu-workgrpoups", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc(
          "Specifies the number of workgroups to use for GPU dispatch")};
  Option<bool> tiledLoopToAsync{
      *this, "convert-to-async",
      llvm::cl::desc("Convert top level tiled.loop op to async op"),
      llvm::cl::init(false)};
  Option<bool> distributeTiledLoopToGPUsIds{
      *this, "distribute-to-gpu-ids",
      llvm::cl::desc("Distribute tiled loop on gpu blocks"),
      llvm::cl::init(false)};
  Option<bool> tiledLoopToSCF{
      *this, "tiled-loop-to-scf",
      llvm::cl::desc("Lower tiled.loop ops to scf.for."),
      llvm::cl::init(false)};
};
}  // end anonymous namespace

static void runStrategy(LinalgTensorCodegenStrategyPass &pass,
                        CodegenStrategy &strategy) {
  strategy.setEnableLICM(pass.licm)
      .setEnableHoistRedundantVectorTransfers(
          pass.hoistRedundantVectorTransfers)
      .setEnableHoistRedundantVectorTransfersOnTensor(
          pass.hoistRedundantVectorTransfers)
      .setEnableVectorTransferPartialRewrite(pass.vectorTransferPartialRewrite)
      .setEnableVectorContractLowering(pass.vectorContractLowering)
      .setEnableVectorToSCFConversion(pass.vectorToSCFConversion)
      .transform(pass.getFunction());
}

static void runGenericStrategy(
    LinalgTensorCodegenStrategyPass &pass, LinalgTilingOptions tilingOptions,
    vector::VectorContractLowering vectorContractLowering,
    vector::VectorTransferSplit vectorTransferSplit) {
  assert(!pass.anchorOpName.empty());
  CodegenStrategy strategy;
  strategy
      .tileIf<LinalgOp>(!pass.tileSizes.empty(), pass.anchorOpName,
                        tilingOptions)
      .vectorizeIf(pass.vectorize, pass.anchorOpName)
      .setVectorTransformsOptions(
          vector::VectorTransformsOptions()
              .setVectorTransformsOptions(vectorContractLowering)
              .setVectorTransferSplit(vectorTransferSplit))
      .setVectorTransferToSCFOptions(
          VectorTransferToSCFOptions().setUnroll(pass.unrollVectorTransfers));
  runStrategy(pass, strategy);
}

template <typename OpType>
static void runStrategy(LinalgTensorCodegenStrategyPass &pass,
                        LinalgTilingOptions tilingOptions,
                        vector::VectorContractLowering vectorContractLowering,
                        vector::VectorTransferSplit vectorTransferSplit) {
  CodegenStrategy strategy;
  strategy.tileIf<OpType>(!pass.tileSizes.empty(), tilingOptions)
      .template vectorizeIf<OpType>(pass.vectorize)
      .setVectorTransformsOptions(
          vector::VectorTransformsOptions()
              .setVectorTransformsOptions(vectorContractLowering)
              .setVectorTransferSplit(vectorTransferSplit))
      .setVectorTransferToSCFOptions(
          VectorTransferToSCFOptions().setUnroll(pass.unrollVectorTransfers));
  runStrategy(pass, strategy);
}

template <typename IdOpTy, typename CountOpTy, char dim>
static linalg::ProcInfo getGpuProcInfo(OpBuilder &b, Location loc) {
  Type indexType = b.getIndexType();
  std::string d(1, dim);
  StringAttr attr = b.getStringAttr(d);
  ProcInfo procInfo = {b.create<IdOpTy>(loc, indexType, attr),
                       b.create<CountOpTy>(loc, indexType, attr)};
  return procInfo;
}

static LinalgLoopDistributionOptions getDistributionOptions() {
  LinalgLoopDistributionOptions opts;
  opts.procInfoMap.insert(std::make_pair(
      "block_x", getGpuProcInfo<gpu::BlockIdOp, gpu::GridDimOp, 'x'>));
  opts.procInfoMap.insert(std::make_pair(
      "block_y", getGpuProcInfo<gpu::BlockIdOp, gpu::GridDimOp, 'y'>));
  opts.procInfoMap.insert(std::make_pair(
      "thread_x", getGpuProcInfo<gpu::ThreadIdOp, gpu::BlockDimOp, 'x'>));
  opts.procInfoMap.insert(std::make_pair(
      "thread_y", getGpuProcInfo<gpu::ThreadIdOp, gpu::BlockDimOp, 'y'>));
  return opts;
}

// For now, just assume it is the zero of type.
// In the future, it should be the zero of type + op.
static Value getNeutralOfLinalgOp(OpBuilder &b, OpOperand &op) {
  auto t = getElementTypeOrSelf(op.get().getType());
  return b.create<ConstantOp>(op.getOwner()->getLoc(), t, b.getZeroAttr(t));
}

namespace {
/// Canonicalize AffineMinOp operations in the context of tiled loops with a
/// known range.
struct AffineMinTiledLoopCanonicalizationPattern
    : public OpRewritePattern<AffineMinOp> {
  using OpRewritePattern<AffineMinOp>::OpRewritePattern;

  // getTiledLoopOpMinMax
  LogicalResult matchAndRewrite(AffineMinOp minOp,
                                PatternRewriter &rewriter) const override {
    return scf::canonicalizeMinMaxOpInLoop(
        rewriter, minOp, minOp.map(), minOp.operands(), /*isMin=*/true,
        /*loopMatcher=*/[](Value iv, Value &lb, Value &ub, Value &step) {
          auto ivArg = iv.dyn_cast<BlockArgument>();
          if (!ivArg) return failure();
          auto tiledLoopOp =
              dyn_cast<TiledLoopOp>(ivArg.getOwner()->getParentOp());
          if (!tiledLoopOp) return failure();
          unsigned idx = ivArg.getArgNumber();
          lb = tiledLoopOp.lowerBound()[idx];
          ub = tiledLoopOp.upperBound()[idx];
          step = tiledLoopOp.step()[idx];
          return success();
        });
  }
};
}  // namespace

/// Apply transformations specified as patterns.
void LinalgTensorCodegenStrategyPass::runOnFunction() {
  auto funcOp = getFunction();
  if (!anchorFuncOpName.empty() && anchorFuncOpName != funcOp.getName()) return;

  if (fuse) {
    // Collect all Linalg ops, they must all have tensor semantics.
    // For now this just fuses everything.
    // TODO: finer control.
    SmallVector<LinalgOp> linalgOps;
    auto walkResult = funcOp.walk([&](LinalgOp op) {
      if (!op.hasTensorSemantics()) return WalkResult::interrupt();
      linalgOps.push_back(op);
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) return signalPassFailure();

    linalg::Aliases aliases;
    LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
    OpBuilder builder(funcOp.getContext());
    linalg::LinalgTilingLoopType loopType = LinalgTilingLoopType::Loops;
    LinalgTilingOptions tilingOptions;
    tilingOptions = tilingOptions.setTileSizes(tileSizes).setLoopType(loopType);
    if (fusePadding) {
      tilingOptions = tilingOptions.setPaddingValueComputationFunction(
          getNeutralOfLinalgOp);
      llvm_unreachable("NYI: fusion does not call padding pattern");
    }
    Optional<TiledAndFusedLinalgOps> tileAndFuseOps = tileAndFuseLinalgOps(
        builder, linalgOps, dependenceGraph, tilingOptions);
    if (tileAndFuseOps)
      linalgOps.back().getOperation()->replaceAllUsesWith(
          tileAndFuseOps->fusedLoops.front());
  }

  if (distribute && !distributeTileSizes.empty()) {
    SmallVector<StringRef, 2> distributionTypesRefs(distributionTypes.begin(),
                                                    distributionTypes.end());
    LinalgTilingOptions tilingOptions;
    tilingOptions.setLoopType(LinalgTilingLoopType::TiledLoops)
        .setTileSizes(distributeTileSizes)
        .setDistributionTypes(distributionTypesRefs);
    if (pad)
      tilingOptions = tilingOptions.setPaddingValueComputationFunction(
          getNeutralOfLinalgOp);
    OwningRewritePatternList patterns(funcOp.getContext());

    populateTileAndFusePattern(
        patterns, tilingOptions,
        LinalgTransformationFilter(
            ArrayRef<Identifier>{},
            {Identifier::get("distributed", funcOp.getContext())})
            .addFilter([](Operation *op) {
              return success(isaContractionOpInterface(dyn_cast<LinalgOp>(op)));
            }));
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    // Ensure we drop the marker in the end.
    funcOp.walk([](LinalgOp op) {
      op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });
    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(funcOp.getContext());
    canonicalizationPatterns.add<AffineMinTiledLoopCanonicalizationPattern>(
        funcOp.getContext());
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(canonicalizationPatterns));
  }

  LinalgTilingOptions tilingOptions;
  if (!tileSizes.empty()) tilingOptions = tilingOptions.setTileSizes(tileSizes);
  if (pad)
    tilingOptions =
        tilingOptions.setPaddingValueComputationFunction(getNeutralOfLinalgOp);

  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          vectorizeContractionTo.getValue())
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  if (!anchorOpName.empty()) {
    using RunFnType = decltype(&runGenericStrategy);
    RunFnType runFn = &runGenericStrategy;
    // `linalg::PadTensorOp::getOperationName()` is not a StringLiteral, cannot
    // use StringSwitch.
    if (anchorOpName == CopyOp::getOperationName())
      runFn = &runStrategy<CopyOp>;
    if (anchorOpName == FillOp::getOperationName())
      runFn = &runStrategy<FillOp>;
    if (anchorOpName == PadTensorOp::getOperationName())
      runFn = &runStrategy<PadTensorOp>;
    else if (anchorOpName == MatmulOp::getOperationName())
      runFn = &runStrategy<MatmulOp>;
    else if (anchorOpName == MatmulI8I8I32Op::getOperationName())
      runFn = &runStrategy<MatmulI8I8I32Op>;
    runFn(*this, tilingOptions, vectorContractLowering, vectorTransferSplit);
  }

  // Transforms that do not require anchoring on a given op.
  if (hoistPadding > 0) {
    SmallVector<PadTensorOp> ops;
    funcOp.walk([&](PadTensorOp op) { ops.push_back(op); });
    for (auto op : llvm::reverse(ops))
      (void)hoistPaddingOnTensors(op, hoistPadding);
  }
  if (vectorizePadding) {
    OwningRewritePatternList extraVectorizationPatterns(funcOp.getContext());
    populatePadTensorOpVectorizationPatterns(extraVectorizationPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(extraVectorizationPatterns));
  }
  if (tiledLoopToGPU && !numberGPUWorkgroups.empty()) {
    OwningRewritePatternList tiledLoopsToGPUPatterns(funcOp.getContext());
    populateTiledLoopsToGPUPatterns(tiledLoopsToGPUPatterns,
                                    numberGPUWorkgroups);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(tiledLoopsToGPUPatterns));
  }
  if (tiledLoopToAsync) {
    OwningRewritePatternList tiledLoopsToAsyncPatterns(funcOp.getContext());
    populateTiledLoopToAsyncPatterns(tiledLoopsToAsyncPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(tiledLoopsToAsyncPatterns));
  }
  if (distributeTiledLoopToGPUsIds) {
    OwningRewritePatternList distributeTiledLoopsPatterns(funcOp.getContext());
    // Distribute the inner loop on threads and the outter loop on blocks.
    populateLinalgDistributeTiledLoopPattern(
        distributeTiledLoopsPatterns, getDistributionOptions(),
        LinalgTransformationFilter(
            ArrayRef<Identifier>{},
            {Identifier::get("distributed", funcOp.getContext())})
            .addFilter([](Operation *op) {
              return success(!op->getParentOfType<linalg::TiledLoopOp>());
            }));
    populateLinalgDistributeTiledLoopPattern(
        distributeTiledLoopsPatterns, getDistributionOptions(),
        LinalgTransformationFilter(
            ArrayRef<Identifier>{},
            {Identifier::get("distributed", funcOp.getContext())})
            .addFilter([](Operation *op) {
              return success(op->getParentOfType<linalg::TiledLoopOp>());
            }));
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(distributeTiledLoopsPatterns));
    // Ensure we drop the marker in the end.
    funcOp.walk([](TiledLoopOp op) {
      op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });
  }
  if (tiledLoopToSCF) {
    OwningRewritePatternList tiledLoopsToSCFPatterns(funcOp.getContext());
    populateTiledLoopToSCFPattern(tiledLoopsToSCFPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(tiledLoopsToSCFPatterns));
  }
}

namespace mlir {
namespace linalg {
void registerLinalgTensorCodegenStrategyPass() {
  PassRegistration<LinalgTensorCodegenStrategyPass>
      testLinalgCodegenStrategyPass;
}
}  // namespace linalg
}  // namespace mlir
