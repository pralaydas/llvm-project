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
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
//===----------------------------------------------------------------------===//
// Include header file for Poseidon dialect
//===----------------------------------------------------------------------===//
#include "Poseidon/PoseidonDialect.h"
#include "Poseidon/PoseidonOps.h"
#include "Poseidon/Passes.h"



using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLinalg RewritePatterns
//===----------------------------------------------------------------------===//
Value getDimOp(OpBuilder &b, Location loc, Value v, int dim) {
  return b.createOrFold<tensor::DimOp>(loc, v, dim);
}
SmallVector<Value> getTensorSizesUntilDim(OpBuilder &b, Location loc,
                                          Value tensor, int dim) {
  RankedTensorType type = tensor.getType().cast<RankedTensorType>();
  assert(dim < type.getRank() &&
         "The given dim must be smaller than tensor rank");
  (void)type;
  SmallVector<Value> sizes;
  for (int i = 0; i <= dim; i++)
    sizes.push_back(getDimOp(b, loc, tensor, i));
  return sizes;
}

SmallVector<Value> getTensorSizes(OpBuilder &b, Location loc, Value tensor) {
  RankedTensorType type = tensor.getType().cast<RankedTensorType>();
  return getTensorSizesUntilDim(b, loc, tensor, type.getRank() - 1);
}
Value createZeroInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                           Type elemTy) {
  Value initTensor =
      b.create<tensor::EmptyOp>(loc, getAsOpFoldResult(sizes), elemTy);
  RankedTensorType type = initTensor.getType().cast<RankedTensorType>();
  Value c0 =
      b.create<arith::ConstantOp>(loc, b.getZeroAttr(type.getElementType()));
  return b.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);
}
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
                diag << "expected 'main' to have 0 inputs and 0 results";
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
        // ElementsAttr value = op.getValue().cast<ElementsAttr>();

        // auto value = op.getValue();
        // DenseElementsAttr attr = op.getValueAttr();
        // llvm::errs()<<op.getType().getElementType()<<"\n";
        
        
        // ArrayRef<float> val;
        // llvm::errs()<<tensorType.getShape()<<"\n";
        // auto ty = RankedTensorType::get(tensorType.getShape(), rewriter.getF64Type());


        // llvm::errs()<<ty<<"\n";

        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getType(), op.getValue());
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
struct AddopLowering : public OpConversionPattern<toy::AddOp> {
    using OpConversionPattern<toy::AddOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(toy::AddOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const final{
        Location loc = op.getLoc();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();

        // auto tensorType = op.getType().cast<TensorType>();
        // auto memRefType = convertTensorToMemRef(tensorType);
        // auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
        // rewriter.replaceOp(op,alloc);

        
        auto lhsType = lhs.getType().cast<RankedTensorType>();
        auto rhsType = rhs.getType().cast<RankedTensorType>();

        // Get the rank of both matrix.
        unsigned lhsRank = lhsType.getRank();
        unsigned rhsRank = rhsType.getRank();

        // Type newResultType = getTypeConverter()->convertType(op.getType());
        // auto resultType = newResultType.cast<RankedTensorType>();
        // Type elementType = resultType.getElementType();
        Type elementType = lhs.getType().cast<RankedTensorType>().getElementType();
        // Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
        // Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);

        // Value initTensor = createZeroInitTensor(
        // rewriter, loc, getTensorSizes(rewriter, loc, lhs), elementType);


        
        // Value initTensor = rewriter.create<linalg::initTensor>(loc, ValueRange{lhsDim0});
        SmallVector<AffineMap,2> indexingMaps = {
            rewriter.getMultiDimIdentityMap(1),
            rewriter.getMultiDimIdentityMap(1),
            rewriter.getMultiDimIdentityMap(1)
        };
        SmallVector<utils::IteratorType> iteratorTypes(
          lhsRank, utils::IteratorType::parallel);
        
        
        Value addop = rewriter.create<linalg::GenericOp>(
                loc, lhs.getType(),  
                /*inputs=*/ValueRange{lhs,rhs}, 
                /*outputs=*/lhs, 
                /*indexingMaps=*/indexingMaps,
                /*iterationTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value input_lhs = args[0];
                  Value input_rhs = args[1];
                  Value output = args[2];
                  Value result = b.create<arith::AddFOp>(loc, input_lhs, input_rhs);
                //   Value result1 = b.create<arith::AddFOp>(loc, output, result);
                  b.create<linalg::YieldOp>(loc, result);
                }).getResult(0);
        // llvm::errs()<<addop<<"\n";
        // rewriter.replaceOpWithNewOp<tensor::CastOp>(op, lhs.getType(), addop);
        rewriter.replaceOpWithNewOp<arith::AddFOp>(op, op.getOperand(0), op.getOperand(1));
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
struct ToyToLinalgLoweringPass
    : public PassWrapper<ToyToLinalgLoweringPass, OperationPass<ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLinalgLoweringPass)

        void getDependentDialects(DialectRegistry &registry) const override {
            registry.insert<poseidon::PoseidonDialect>();
            registry.insert<AffineDialect, func::FuncDialect, 
            memref::MemRefDialect, linalg::LinalgDialect,
            tosa::TosaDialect>();
        }
        void runOnOperation() override;
};
} // namespace

void ToyToLinalgLoweringPass::runOnOperation() {
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
            linalg::LinalgDialect, tosa::TosaDialect>();

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
    
    // populateToyToLinalgConversionPatterns(typeConverter, patterns);
    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if(failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

/// Create a pass for lowering operations in the `Poseidon` dialect,
/// for a subset of the Toy IR (e.g. constant & return).

std::unique_ptr<Pass> mlir::poseidon::createLowerToLinalgPass() {
    return std::make_unique<ToyToLinalgLoweringPass>();
}