//====- LowerToPoseidonLoops.cpp - Partial lowering from Toy to Poseidon --===//
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
using namespace mlir;




//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  // ArrayRef<int64_t> steps(tensorType.getRank(), 1);
  // llvm::errs()<<tensorType.getRank()<<"\n";
  // llvm::errs() <<tensorType.getShape() <<"\n";
  buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        
        auto valueToStore = processIteration(nestedBuilder, operands, ivs);
        // nestedBuilder.create<AffineStoreOp>(loc, valueToStore, alloc, ivs);
        nestedBuilder.create<AffineVectorStoreOp>(loc, valueToStore, alloc, ivs);
      });
  
  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>

struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    
    // auto vecType = (*op->operand_type_begin());
    mlir::MLIRContext *context = op->getContext();
    Type elementType = mlir::FloatType::getF64(context);
    int64_t numElements = 16;
    VectorType vecType = VectorType::get(numElements, elementType);

    lowerOpToLoops(op, operands, rewriter,
                   [loc, vecType](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the
                     // BinaryOp. This allows for using the nice named accessors
                     // that are generated by the ODS.
                     typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);
                     //  llvm::errs() << binaryAdaptor <<"\n";
                     // Generate loads for the element of 'lhs' and 'rhs' at the
                     // inner loop.
                    //  auto loadedLhs = builder.create<AffineLoadOp>(
                    //      loc, binaryAdaptor.getLhs(), loopIvs);
                    //  auto loadedRhs = builder.create<AffineLoadOp>(
                    //      loc, binaryAdaptor.getRhs(), loopIvs);
                    // auto inElemTy = operands[0].getType().cast<MemRefType>().getElementType()
                    // VectorType vectorTy32 = VectorType::get({2}, inElemTy);
                    
                     auto loadedLhs = builder.create<AffineVectorLoadOp>(
                        loc, vecType, binaryAdaptor.getLhs(), loopIvs);
                    auto loadedRhs = builder.create<AffineVectorLoadOp>(
                        loc, vecType, binaryAdaptor.getRhs(),loopIvs);
                    // auto loadedLhs = builder.create<vector::TransferReadOp>(
                    //     loc, binaryAdaptor.getLhs(), loopIvs);
                    // auto loadedRhs = builder.create<vector::TransferReadOp>(
                    //     loc, binaryAdaptor.getRhs(),loopIvs);
                     // Create the binary operation performed on the loaded
                     // values.
                     
                     return builder.create<LoweredBinaryOp>(loc, vecType, loadedLhs,
                                                            loadedRhs);
                   });
    
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, poseidon::Addop>;
// using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
// using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;


inline static Type cvtTensorToMemref(Type type) {
    auto tt = type.cast<TensorType>();
    return MemRefType::get(tt.getShape(), tt.getElementType());
}

template <class Op>
struct LowerOp : public OpRewritePattern<Op> {
    LowerOp(MLIRContext *ctx) : OpRewritePattern<Op>(ctx) {}

    LogicalResult matchAndRewrite(Op op,
                                  PatternRewriter &rewriter) const override;

    virtual LogicalResult lower(Op op, ValueRange buffers,
                                PatternRewriter &rewriter) const = 0;
};

template <class Op>
LogicalResult LowerOp<Op>::matchAndRewrite(Op op,
                                           PatternRewriter &rewriter) const {
    // Find buffers for function outputs
    SmallVector<Value> newResults;
    bool isFuncRet = false;
    
    for (auto result : op->getResults()) {
        // Check if the result of this operation is returned by the parent
        // function
        Value newResult;
        func::ReturnOp retOp;
        int64_t retIdx = -1;
        for (auto &use : result.getUses()) {
            auto owner = use.getOwner();
            if (isa<func::ReturnOp>(owner)) {
                retOp = cast<func::ReturnOp>(owner);
                retIdx = use.getOperandNumber();
                isFuncRet = true;
            }
        }

        // Collect result buffer or allocate a new one
        if (retOp) {
            auto func = cast<func::FuncOp>(op->getParentOp());
            auto numInputs =
                func->template getAttrOfType<IntegerAttr>("num_inputs")
                    .getInt();
            newResult = func.getArgument(numInputs + retIdx);
        } else {
          llvm::errs()<<"*****************************\n";
            auto alloc = rewriter.create<memref::AllocOp>(
                op.getLoc(), result.getType().template cast<MemRefType>());
                llvm::errs()<<"*****************************\n";
            newResult = alloc.getResult();
        }
        newResults.push_back(newResult);
    }

    auto buffers = llvm::to_vector(op->getOperands());
    buffers.append(newResults);

    // Lower operation with given buffers
    if (this->lower(op, buffers, rewriter).failed()) return failure();

    // Erase or replace previous operation
    if (!isFuncRet)
        rewriter.replaceOp(op, newResults);
    else
        rewriter.eraseOp(op);

    return success();
}

#define FOR(iv, low, high, body)                                          \
    auto iv##Loop = rewriter.create<AffineForOp>(op.getLoc(), low, high); \
    rewriter.setInsertionPointToStart(iv##Loop.getBody());                \
    {                                                                     \
        auto iv = iv##Loop.getInductionVar();                             \
        body                                                              \
    }                                                                     \
    rewriter.setInsertionPointAfter(iv##Loop);

#define LOAD(buffer, indices) \
    rewriter.create<AffineLoadOp>(op.getLoc(), buffer, indices)

#define STORE(value, buffer, indices) \
    rewriter.create<AffineStoreOp>(op.getLoc(), value, buffer, indices)

#define F32_CONST(value)                                                   \
    rewriter                                                               \
        .create<arith::ConstantFloatOp>(op.getLoc(), llvm::APFloat(value), \
                                        rewriter.getF32Type())             \
        .getResult()

#define BOP(Op, lhs, rhs) rewriter.create<Op>(op.getLoc(), lhs, rhs).getResult()

#define ADDF(lhs, rhs) BOP(arith::AddFOp, lhs, rhs)
#define MULF(lhs, rhs) BOP(arith::MulFOp, lhs, rhs)
#define MAXF(lhs, rhs) BOP(arith::MaxFOp, lhs, rhs)

inline static void genNestedLoops(
    Value result, PatternRewriter &rewriter,
    function_ref<void(const SmallVector<Value> &)> body) {
    auto shape = result.getType().cast<MemRefType>().getShape();
    SmallVector<Value> ivs;
    for (auto dim : shape) {
        auto loop = rewriter.create<AffineForOp>(result.getLoc(), 0, dim);
        ivs.push_back(loop.getInductionVar());
        rewriter.setInsertionPointToStart(loop.getBody());
    }
    body(ivs);
}

struct AddOPLowering : public LowerOp<toy::AddOp> {
    AddOPLowering(MLIRContext *ctx) : LowerOp(ctx) {}

    LogicalResult lower(toy::AddOp op, ValueRange buffers,
                        PatternRewriter &rewriter) const override {
        auto data = buffers[0], weight = buffers[1], result = buffers[2];
        auto dataShape = data.getType().cast<MemRefType>().getShape();
        auto weightShape = weight.getType().cast<MemRefType>().getShape();
        auto batchSize = dataShape[0], inDim = dataShape[1],
             outDim = weightShape[0];
        
        FOR(i, 0, batchSize,  // for (i, 0, data.shape[0])
            FOR(
                j, 0, outDim,  // for (j, 0, weight.shape[0])
                auto kLoop = rewriter.create<AffineForOp>(
                    op.getLoc(), 0, inDim, 1, ValueRange{F32_CONST(0.f)});
                rewriter.setInsertionPointToStart(kLoop.getBody()); {
                    auto k = kLoop.getInductionVar();
                    auto D_ik = LOAD(data, (ValueRange{i, k}));
                    auto W_jk = LOAD(weight, (ValueRange{j, k}));
                    // auto mul = MULF(D_ik, W_jk);
                    // auto prev = kLoop.getRegionIterArgs()[0];
                    // auto add = ADDF(prev, mul);
                    auto add = ADDF(D_ik, W_jk);
                    rewriter.create<AffineYieldOp>(op.getLoc(),
                                                   ValueRange{add});
                } rewriter.setInsertionPointAfter(kLoop);
                STORE(kLoop->getResult(0), result,
                      (ValueRange{i, j}));)  // end j
            )                                // end i

        return success();
    }
};


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

// struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
//   using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(toy::ConstantOp op,
//                                 PatternRewriter &rewriter) const override {
    
//     DenseElementsAttr constantValue = op.getValue();
//     Location loc = op.getLoc();

//     // When lowering the constant operation, we allocate and assign the constant
//     // values to a corresponding memref allocation.
//     auto tensorType = op.getType().cast<TensorType>();
//     auto memRefType = convertTensorToMemRef(tensorType);
//     auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
    
    

//     // Value newOp = rewriter.create<poseidon::Constantop>(op, memRefType, op.getValue());
//     // rewriter.replaceOpWithNewOp<poseidon::Constantop>(op, memRefType, op.getValue());
//     // rewriter.replaceOpWithNewOp<poseidon::Constantop>(op, memRefType, constantValue);
    
//     rewriter.create<memref::TensorStoreOp>(
//       loc, 
//       // rewriter.replaceOpWithNewOp<poseidon::Constantop>(op, tensorType, constantValue),
//       rewriter.create<poseidon::Constantop>(loc, constantValue),
//       alloc);

//     rewriter.replaceOp(op, alloc);
//     return success();
//   }
// };
//===----------------------------------------------------------------------===//
// ToyToMemref RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

class ConstantOpLowering : public ConversionPattern {
public:
  ConstantOpLowering(MLIRContext *context)
      : ConversionPattern(toy::ConstantOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    
    
    auto loc = op->getLoc();
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    
    auto memRefType = convertTensorToMemRef(tensorType);
    auto allocOp = insertAllocAndDealloc(memRefType, loc, rewriter);
    
    auto constantOp = cast<toy::ConstantOp>(op);

    // Get the tensor attribute from the toy.constant operation.
    DenseElementsAttr tensorAttr = constantOp.getValue();
    
    auto attrValue = op->getAttrs();
    
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto *context = parentModule.getContext();
    auto *context_ = op->getContext();
    
    std::string pega  = "constant_";

    static int pos_val = 0;
    pega  = pega + std::to_string(pos_val);
    pos_val ++;

    getOrCreateGlobal(
        loc, rewriter, pega, tensorAttr, parentModule, memRefType, op);
    
    auto getglobalOp = rewriter.create<memref::GetGlobalOp>(loc, memRefType,
        mlir::FlatSymbolRefAttr::get(context_, pega));

    
    // auto allocOp  = rewriter.create<memref::AllocOp>(loc, memRefType);
    auto copyOp = rewriter.create<memref::CopyOp>(loc, getglobalOp, allocOp);

    // rewriter.eraseOp(op);
    rewriter.replaceOp(op,allocOp);
    return success();
  }
  private:
  static void getOrCreateGlobal(Location loc, OpBuilder &builder,
                                       StringRef name, DenseElementsAttr tensorAttr,
                                       ModuleOp module, MemRefType memRefType, Operation *op){
    
    auto *context = module.getContext();
    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    
    builder.setInsertionPointToStart(module.getBody());
    
    auto globalop = builder.create<memref::GlobalOp>(
          loc, 
          mlir::StringAttr::get(context, name),
          mlir::StringAttr::get(context, "private"),
          mlir::TypeAttr::get(memRefType),
          tensorAttr,
          mlir::UnitAttr::get(context),
          nullptr);

    return;    
  }
};
//===----------------------------------------------------------------------===//
// ToyToArith RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//


// class ConstantOpLowering : public ConversionPattern {
// public:
//   explicit ConstantOpLowering(MLIRContext *context)
//       : ConversionPattern(toy::ConstantOp::getOperationName(), 1, context) {}

//   // Specify the conversion pattern.
//   LogicalResult
//   matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto toyConstOp = cast<toy::ConstantOp>(op);

    
//     Location loc = op->getLoc();
//     auto tensorType = toyConstOp.getType().cast<TensorType>();
    
//     auto memRefType = convertTensorToMemRef(tensorType);
    
//     auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
//     // Convert the constant to an arith.constantop operation.
//     // auto arithConstOp =
//     //     rewriter.create<arith::ConstantOp>(op->getLoc(), toyConstOp.getValue());

//     // Replace the toy.constantop with the arith.constantop.
//     // rewriter.replaceOp(op, arithConstOp.getResult());
//     rewriter.create<memref::TensorStoreOp>(
//       loc, 
//       // rewriter.replaceOpWithNewOp<poseidon::Constantop>(op, tensorType, constantValue),
//       rewriter.create<arith::ConstantOp>(op->getLoc(), toyConstOp.getValue()),
//       alloc);

//     rewriter.replaceOp(op, alloc);
//     return success();
//   }
// };


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
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
// ToyToAffine RewritePatterns: Print operations
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

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> {
  using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "toy.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

} // namespace


//===----------------------------------------------------------------------===//
// ToyToPoseidonLoopsLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToPoseidonLoopsLoweringPass
    : public PassWrapper<ToyToPoseidonLoopsLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToPoseidonLoopsLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<poseidon::PoseidonDialect, func::FuncDialect, memref::MemRefDialect, 
                    arith::ArithDialect, AffineDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;
};
} // namespace

void ToyToPoseidonLoopsLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `poseidon`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<poseidon::PoseidonDialect, AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                         vector::VectorDialect>();

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
  patterns.add< AddOpLowering, ConstantOpLowering,FuncOpLowering, ReturnOpLowering,
                 PrintOpLowering >(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::poseidon::createLowerToPoseidonLoopsPass() {
  return std::make_unique<ToyToPoseidonLoopsLoweringPass>();
}
