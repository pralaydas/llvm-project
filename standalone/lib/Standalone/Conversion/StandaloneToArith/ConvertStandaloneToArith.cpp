#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "Standalone/StandalonePasses.h"

using namespace mlir;
// using namespace mlir::memref;
// using namespace mlir::standalone;

namespace {


//===----------------------------------------------------------------------===//
// Standalone to arith RewritePatterns: Add operations
//===----------------------------------------------------------------------===//

struct AddOpLowering : public OpRewritePattern<standalone::AddOp> {
  using OpRewritePattern<standalone::AddOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(standalone::AddOp op,
                                PatternRewriter &rewriter) const override {
    
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, op.getOperand(0), op.getOperand(1));
    return success();
  }
};
} // namespace

// find the ops that can be converted
namespace mlir {
void populateStandaloneToArithConversionPatterns(MLIRContext *context,
                                                 RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<AddOpLowering>(context);
  // clang-format on
}
} // namespace mlir
  //
namespace {
struct StandaloneToArithLoweringPass
    : public PassWrapper<StandaloneToArithLoweringPass, ::mlir::OperationPass<::mlir::func::FuncOp>> {
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect>();
  }
  void runOnOperation() override;
  StringRef getArgument() const final { return "standalone-to-arith"; }
  StringRef getDescription() const final {
    return "lower arith dialect to Standalone dialect";
  }
};
} // namespace

// lower all functions
void StandaloneToArithLoweringPass::runOnOperation() {
  ConversionTarget target(getContext());
  
  target.addLegalDialect<func::FuncDialect, arith::ArithDialect>();
  RewritePatternSet patterns(&getContext());
  target.addIllegalDialect<standalone::StandaloneDialect>();
  populateStandaloneToArithConversionPatterns(&getContext(), patterns);
  if(failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

void standalone::createLowerStandaloneToArithPass() {
  PassRegistration<StandaloneToArithLoweringPass>();
}
