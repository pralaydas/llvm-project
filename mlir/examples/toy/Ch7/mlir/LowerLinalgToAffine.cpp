//====- LowerToLinalg.cpp - Partial lowering from Toy to Linalg --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to Linalg op.
// This lowering expects that all calls have been inlined, and all shapes have 
// been resolved.
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
// Include header file for Linalg dialect
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
//===----------------------------------------------------------------------===//
// Include header file for Poseidon dialect
//===----------------------------------------------------------------------===//
#include "Poseidon/PoseidonDialect.h"
#include "Poseidon/PoseidonOps.h"
#include "Poseidon/Passes.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLinalg RewritePatterns
//===----------------------------------------------------------------------===//
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}
namespace {
//===----------------------------------------------------------------------===//
// ToyToPoseidon RewritePatterns: Return operations
//===----------------------------------------------------------------------===//




//===----------------------------------------------------------------------===//
// ToyToPoseidon RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//
struct AddopLowering : public OpConversionPattern<linalg::GenericOp> {
    using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const final{
        
        Location loc = op.getLoc();
        auto inputs = op.getInputs();
        Value input0 = inputs[0];
        Value input1 = inputs[1];
        Value output = op.getOutputs()[0];

        auto input0Type = input0.getType().dyn_cast<TensorType>();
        auto input0MemRefType = convertTensorToMemRef(input0Type);
        ArrayRef<int64_t> input0Shape = input0MemRefType.getShape();
        auto input1Type = input1.getType().dyn_cast<TensorType>();
        auto input1MemRefType = convertTensorToMemRef(input1Type);
        ArrayRef<int64_t> input1Shape = input1MemRefType.getShape();
        auto outputType = output.getType().dyn_cast<TensorType>();
        auto outputMemRefType = convertTensorToMemRef(outputType);
        ArrayRef<int64_t> outputShape = outputMemRefType.getShape();
        // MemRefType input1Type = input1.getType().dyn_cast<MemRefType>();
        // ArrayRef<int64_t> input1Shape = input1Type.getShape();
        // MemRefType outputType = output.getType().dyn_cast<MemRefType>();
        // ArrayRef<int64_t> outputShape = outputType.getShape();
        

        llvm::errs()<<input0Shape[0]<<"\n";
        // llvm::errs()<<input1Type<<"\n";
        // llvm::errs()<<outputType<<"\n";
        // rewriter.replaceOpWithNewOp<tensor::CastOp>(op, lhs.getType(), addop.getResult(0));
        // rewriter.replaceOpWithNewOp<tensor::CastOp>(op, lhs.getType(), addop);
        // rewriter.replaceOpWithNewOp<arith::AddFOp>(op, op.getOperand(0), op.getOperand(1));
        // rewriter.eraseOp(op);
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
    llvm::errs()<<"***********************\n";
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};
} //namespace

// void populateToyToLinalgConversionPatterns(TypeConverter &converter, 
//                                     RewritePatternSet &patterns) {
//   // clang-format off
//   patterns.add<AddopLowering>(converter);
//   // clang-format on
// }


//===----------------------------------------------------------------------===//
// ToyToPoseidonLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Poseidon of the toy operations that are
/// computationally intensive (like constant, matmul for example...) while keeping the
/// rest of the code in the Toy dialect.

namespace{
struct LinalgToAffineLoweringPass
    : public PassWrapper<LinalgToAffineLoweringPass, OperationPass<ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToAffineLoweringPass)

        void getDependentDialects(DialectRegistry &registry) const override {
            registry.insert<poseidon::PoseidonDialect>();
            registry.insert<AffineDialect, func::FuncDialect, 
            memref::MemRefDialect, vector::VectorDialect>();
        }
        void runOnOperation() override;
};
} // namespace

void LinalgToAffineLoweringPass::runOnOperation() {
    //The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    MLIRContext *context = &getContext();
    ConversionTarget target(getContext());
    TypeConverter typeConverter;
    
    // RewritePatternSet patterns1(context);
    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to our custom Poseidon dialect.
    target.addLegalDialect<poseidon::PoseidonDialect, 
            func::FuncDialect, 
            memref::MemRefDialect,
            AffineDialect, BuiltinDialect, arith::ArithDialect,
            vector::VectorDialect>();

    // We also define the Toy dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the Toy operations that don't want
    // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
    // to be updated though (as we convert from TensorType to MemRefType), so we
    // only treat it as `legal` if its operands are legal.
    target.addIllegalDialect<linalg::LinalgDialect>();
    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
        return llvm::none_of(op->getOperandTypes(),
                            [](Type type) { return type.isa<TensorType>(); });
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<PrintOpLowering >(&getContext());
    mlir::linalg::populateLinalgToStandardConversionPatterns(patterns);
    mlir::linalg::populateElementwiseToLinalgConversionPatterns(patterns);
    
    // populateToyToLinalgConversionPatterns(typeConverter, patterns);
    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if(failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

/// Create a pass for lowering operations in the `Poseidon` dialect,
/// for a subset of the Toy IR (e.g. constant & return).

std::unique_ptr<Pass> mlir::poseidon::createLowerLinalgToAffinePass() {
    return std::make_unique<LinalgToAffineLoweringPass>();
}