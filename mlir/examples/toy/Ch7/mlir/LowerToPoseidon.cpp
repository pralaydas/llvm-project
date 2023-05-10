//====- LowerToPoseidon.cpp - Partial lowering from Toy to Affine+Std --===//
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

//===----------------------------------------------------------------------===//
// ToyToPoseidon RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<toy::returnOp> {
    using OprewritePattern<toy::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(toy::ReturnOp op,
                                    PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if(op.hasOperand()) 
        return failure();  

    // we lower 'toy.return' directly to "func.return".
    rewriter.replaceopWithNewOp<func::ReturnOp>(op);
    return success(); 
    }
};


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
            registry.insert<PoseidonDialect> ();
        }
        void runOnOperation() final;
};
} // namespace

void ToyToPoseidonLoweringpass::runOnOperation() {
    //The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());
    
    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to our custom Poseidon dialect.
    target.addLegalDialect<PoseidonDialect, func::FuncDialect>();

    // We also define the Toy dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the Toy operations that don't want
    // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
    // to be updated though (as we convert from TensorType to MemRefType), so we
    // only treat it as `legal` if its operands are legal.
    target.addIllegalDialect<toy::ToyDialect>();
    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op){
        return llvm::none_of(op->getOperandTypes(),
                            [](Type type){ return type.isa<tensorType>(); });
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering, ReturnOpLowering>(&getContext());

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