//====- LowerToPoseidon.cpp - Partial lowering from Toy to Poseidon --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a new dialect 
// named Poseidon. This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
//===----------------------------------------------------------------------===//
// Include header file for Poseidon dialect
//===----------------------------------------------------------------------===//
#include "Poseidon/PoseidonDialect.h"
#include "Poseidon/PoseidonOps.h"
#include "Poseidon/Passes.h"


using namespace mlir;
// using namespace poseidon;


namespace {
//===----------------------------------------------------------------------===//
// ToyToPoseidon RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnopLowering : public OpRewritePattern<toy::ReturnOp> {
    using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(toy::ReturnOp op,
                                    PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if(op.hasOperand()) 
        return failure();  

    // we lower 'toy.return' directly to "func.return".
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success(); 
    }
};

//===----------------------------------------------------------------------===//
// ToyToPoseidon RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncopLowering : public OpConversionPattern<toy::FuncOp> {
    using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

    
    LogicalResult matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final {
        // We only lower the main function as we expect that all other functions
        // have been inlined.
        if (op.getName() != "main")
            return failure();
        
        // Verify that the given main has no inputs and results.
        if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
            return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
                diag << "eexpected 'main' to have 0 inputs and 0 results";
            });
        }
        // Create a new non-toy function, with the same region.
        auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                        op.getFunctionType());
        rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
        rewriter.eraseOp(op);
        return success();
    }
};


//===----------------------------------------------------------------------===//
// ToyToPoseidon RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//
struct ConstantopLowering : public OpRewritePattern<toy::ConstantOp> {
    using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(toy::ConstantOp op,
                                    PatternRewriter &rewriter) const override {

        // Location loc = op.getLoc();
        // auto tensorType = op.getType().cast<TensorType>();
        // auto memRefType = convertTensorToMemRef(tensorType);
        // auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
        // rewriter.replaceOp(op, alloc);
        rewriter.replaceOpWithNewOp<poseidon::Constantop>(op, op.getType(), op.getValue());
        // auto toyConstOp = cast<toy::ConstantOp>(op);

        // Convert the constant to an arith.constantop operation.
        // auto arithConstOp =
        //     rewriter.create<arith::ConstantOp>(op->getLoc(), toyConstOp.getValue());

        // Replace the toy.constantop with the arith.constantop.
        // rewriter.replaceOp(op, arithConstOp.getResult());
        return success();
    }
};


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//
struct AddopLowering : public OpRewritePattern<toy::AddOp> {
    using OpRewritePattern<toy::AddOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(toy::AddOp op,
                                    PatternRewriter &rewriter) const override {
        // Location loc = op.getLoc();
        // auto tensorType = op.getType().cast<TensorType>();
        // auto memRefType = convertTensorToMemRef(tensorType);
        // auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
        // rewriter.replaceOp(op,alloc);
        rewriter.replaceOpWithNewOp<poseidon::Addop>(op, op.getOperand(0), op.getOperand(1));
        
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToPoseidon RewritePatterns: Print operations
//===----------------------------------------------------------------------===//


struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  
  LogicalResult matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};


} //namespace

//===----------------------------------------------------------------------===//
// ToyToPoseidonLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Poseidon of the toy operations that are
/// computationally intensive (like constant, matmul for example...) while keeping the
/// rest of the code in the Toy dialect.

namespace{
struct ToyToPoseidonLoweringPass
    : public PassWrapper<ToyToPoseidonLoweringPass, OperationPass<ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToPoseidonLoweringPass)

        void getDependentDialects(DialectRegistry &registry) const override {
            registry.insert<poseidon::PoseidonDialect>();
            registry.insert<AffineDialect, func::FuncDialect, memref::MemRefDialect>();
        }
        void runOnOperation() override;
};
} // namespace

void ToyToPoseidonLoweringPass::runOnOperation() {
    //The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());
    
    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to our custom Poseidon dialect.
    target.addLegalDialect<poseidon::PoseidonDialect, 
            func::FuncDialect, 
            memref::MemRefDialect,
            AffineDialect, BuiltinDialect, arith::ArithDialect>();

    // We also define the Toy dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the Toy operations that don't want
    // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
    // to be updated though (as we convert from TensorType to MemRefType), so we
    // only treat it as `legal` if its operands are legal.
    target.addIllegalDialect<toy::ToyDialect>();
    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
        return llvm::none_of(op->getOperandTypes(),
                            [](Type type) { return type.isa<TensorType>(); });
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantopLowering, ReturnopLowering, FuncopLowering,
                    AddopLowering, PrintOpLowering >(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if(failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

/// Create a pass for lowering operations in the `Poseidon` dialect,
/// for a subset of the Toy IR (e.g. constant & return).

std::unique_ptr<Pass> mlir::poseidon::createLowerToPoseidonPass() {
    return std::make_unique<ToyToPoseidonLoweringPass>();
}