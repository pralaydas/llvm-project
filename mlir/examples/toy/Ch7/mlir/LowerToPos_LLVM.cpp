//====- LowerToLLVM.cpp - Lowering from Toy+Affine+Std+Poseidon to LLVM ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements full lowering of Toy & Poseidon operations to LLVM MLIR dialect.
// 'toy.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Poseidon(or Arithmetic) + Affine + SCF + Func dialects to the
// LLVM one:
//
//                          Affine --
//                                  |
//                                  v
//                  Poseidon(or Arithmetic) + Func --> LLVM (Dialect)
//                                  ^
//                                  |
//     'toy.print' --> Loop (SCF) --
//
//===----------------------------------------------------------------------===//


#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
//===----------------------------------------------------------------------===//
// Include header file for Poseidon dialect
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/Passes.h"

#include "Poseidon/PoseidonDialect.h"
#include "Poseidon/PoseidonOps.h"
#include "Poseidon/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include <type_traits>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"


// namespace mlir {
// #define GEN_PASS_DEF_POSEIDONTOLLVMCONVERSIONPASS
// #include "mlir/Conversion/Passes.h.inc"
// } // namespace mlir

using namespace mlir;


//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {
/// Lowers `toy.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(toy::PrintOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

    // Create a loop for each of the dimensions within the shape.
    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody())
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1)
        rewriter.create<func::CallOp>(loc, printfRef,
                                      rewriter.getIntegerType(32), newLineCst);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to printf for the current element of the loop.
    auto printOp = cast<toy::PrintOp>(op);
    auto elementLoad =
        rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
    rewriter.create<func::CallOp>(
        loc, printfRef, rewriter.getIntegerType(32),
        ArrayRef<Value>({formatSpecifierCst, elementLoad}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

/// Lowers `poseidon.constant` to llvm dialect.
// class  ConstantOpLowering : public OpRewritePattern<poseidon::Constantop> {
//   public:
//   using OpRewritePattern<poseidon::Constantop>::OpRewritePattern;

//   LogicalResult
//   matchAndRewrite(poseidon::Constantop op,
//                   PatternRewriter &rewriter) const override{

//         rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, op.getValue());
        
//         return success();
//     }
// };

// struct ConstantOpLowering : public ConvertOpToLLVMPattern<poseidon::Constantop> {
//   using ConvertOpToLLVMPattern<poseidon::Constantop>::ConvertOpToLLVMPattern;

//   LogicalResult
//   matchAndRewrite(poseidon::Constantop op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//         // llvm::errs()<<op.getValue() <<"\n";
//         // return LLVM::detail::oneToOneRewrite(op, LLVM::ConstantOp::getOperationName(),
//         //                                adaptor.getOperands(), op->getAttrs(),
//         //                                *getTypeConverter(), rewriter);  

//         auto type = typeConverter->convertType(op.getResult().getType());
//         if (!type || !LLVM::isCompatibleType(type))
//           return rewriter.notifyMatchFailure(op, "failed to convert result type");

//         auto newOp =
//           rewriter.create<LLVM::ConstantOp>(op.getLoc(), type, op.getValue());
//         for (const NamedAttribute &attr : op->getAttrs()) {
//           if (attr.getName().strref() == "value")
//             continue;
//           newOp->setAttr(attr.getName(), attr.getValue());
//         }
//         rewriter.replaceOpWithNewOp<poseidon::Constantop>(op, type, op.getValue());
//         return success();
//     }
// };
class ConstantOpLowering : public ConversionPattern {
public:
  ConstantOpLowering(MLIRContext *context)
      : ConversionPattern(poseidon::Constantop::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    
    auto constantOp = cast<poseidon::Constantop>(op);
    auto loc = constantOp.getLoc();

    // Get the tensor attribute from the toy.constant operation.
    DenseElementsAttr tensorAttr = constantOp.getValue();

    // Convert the tensor value to an LLVM dialect constant representation.
    Value llvmConstant = convertToLLVMConstant(loc, tensorAttr, rewriter);
    // llvm::errs()<<llvmConstant<<"\n";
    rewriter.replaceOp(op, llvmConstant);
    
    llvm::errs()<<"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";
    return success();
  }

private:
  Value convertToLLVMConstant(Location loc, DenseElementsAttr tensorAttr,
                           ConversionPatternRewriter &rewriter) const {
  
  ShapedType tensorType = tensorAttr.getType().cast<ShapedType>();
  Type elementType = tensorType.getElementType();
  llvm::errs()<<elementType<<"\n";
  llvm::errs()<<"***********************\n";
  // Create an LLVM dialect type corresponding to the element type.
  // Type llvmElementType = typeConverter->convertType(elementType);
  // llvm::errs()<<"#######################\n";
  // llvm::errs()<<llvmElementType<<"\n";
  Type llvmf64 = rewriter.getF64Type();
  // return rewriter.create<LLVM::ConstantOp>(loc, tensorAttr);
  // Type llvmElementType = typeConverter->convertType(elementType);
  
  // Extract the constant values from the dense tensor attribute.
  SmallVector<Attribute, 4> constantValues;
  for (Attribute value : tensorAttr.getValues<Attribute>())
    constantValues.push_back(value);
  for(auto i: constantValues)
    llvm::errs()<<i<<"\n";
  llvm::errs() <<constantValues.size()<<"\n";
  llvm::errs()<<"#######################\n";

  // Create an LLVM dialect constant for each element value.
  SmallVector<Value, 4> llvmConstants;
  for (Attribute constantValue : constantValues) {
    APFloat floatValue = constantValue.cast<FloatAttr>().getValue();
    llvmConstants.push_back(
        rewriter.create<LLVM::ConstantOp>(loc, llvmf64,
                                          rewriter.getFloatAttr(llvmf64, floatValue)));
  }
  for(auto i: llvmConstants)
    llvm::errs()<<i<<"\n";
  llvm::errs() <<llvmConstants.size()<<"\n";
  
  // Create an LLVM dialect array constant with the element constants.
  Type llvmArrayType = LLVM::LLVMArrayType::get(llvmf64, llvmConstants.size());
  llvm::errs()<< llvmArrayType <<"\n";

  Value llvmConstantArray = rewriter.create<LLVM::ConstantOp>(loc, llvmArrayType, llvmConstants);
  llvm::errs()<<llvmConstantArray<<"\n";
  // rewriter.setInsertionPointAfterValue(llvmConstantArray);
  // Operation *llvmConstantArrayOp = llvmConstantArray.getDefiningOp();
  // llvm::errs()<<*llvmConstantArrayOp<<"\n";
  // rewriter.setInsertionPointAfter(llvmConstantArrayOp);
  // for (size_t i = 0; i < llvmConstants.size(); ++i)
  //   rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayType, llvmConstants[i],
  //                                        rewriter.getI64ArrayAttr(i));
  return llvmConstantArray;
  }
};

}

// void populatePoseidonToLLVMConversionPatterns(LLVMTypeConverter &converter, 
//                                     RewritePatternSet &patterns) {
//   // clang-format off
//   patterns.add<ConstantOpLowering>(converter);
//   // clang-format on
// }

namespace {
struct PoseidonToLLVMLoweringPass
    : public PassWrapper<PoseidonToLLVMLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PoseidonToLLVMLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
    }
    void runOnOperation() override {

        LLVMConversionTarget target(getContext());
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalOp<ModuleOp>();

        LLVMTypeConverter typeConverter(&getContext());
        RewritePatternSet patterns(&getContext());
        
        
        
        // populatePoseidonToLLVMConversionPatterns(typeConverter, patterns);

        populateAffineToStdConversionPatterns(patterns);
        populateSCFToControlFlowConversionPatterns(patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        
        // poseidon to LLVM lowering pass
        

        // The only remaining operation to lower from the `toy` dialect, is the
        // PrintOp.
        patterns.add<PrintOpLowering>(&getContext());
        llvm::errs()<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
        patterns.add<ConstantOpLowering>(&getContext());
        llvm::errs()<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
        // We want to completely lower to LLVM, so we use a `FullConversion`. This
        // ensures that only legal operations will remain after the conversion.
        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};
} //namespace


// void mlir::poseidon::PoseidonToLLVMLoweringPass::runOnOperation() 

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::poseidon::createLowerToLLVMPass() {
  return std::make_unique<PoseidonToLLVMLoweringPass>();
}
