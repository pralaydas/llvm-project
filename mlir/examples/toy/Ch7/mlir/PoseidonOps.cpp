//===- PoseidonOps.cpp - Poseidon Dialect Ops -----------------------------*- C++ -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines operations in the Poseidon dialect.
//
//===----------------------------------------------------------------------===//

#include "Poseidon/PoseidonDialect.h"
#include "Poseidon/PoseidonOps.h"


#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"


using namespace mlir;
using namespace poseidon;
// #include "Poseidon/PoseidonDialect.cpp.inc"
#include "Poseidon/PoseidonOps.cpp.inc"


//===----------------------------------------------------------------------===//
// Poseidon Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = type.dyn_cast<FunctionType>()) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}



//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
// void Constantop::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                        double value) {
//   auto dataType = RankedTensorType::get({}, builder.getF64Type());
//   auto dataAttribute = DenseElementsAttr::get(dataType, value);
//   Constantop::build(builder, state, dataType, dataAttribute);
// }


/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult Constantop::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return mlir::failure();

  result.addTypes(value.getType());
  return mlir::success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void Constantop::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}

/// Verify that the given attribute value is valid for the given type.
static mlir::LogicalResult verifyConstantForType(mlir::Type type,
                                                 mlir::Attribute opaqueValue,
                                                 mlir::Operation *op) {
  if (type.isa<mlir::TensorType>()) {
    // Check that the value is an elements attribute.
    auto attrValue = opaqueValue.dyn_cast<mlir::DenseFPElementsAttr>();
    if (!attrValue)
      return op->emitError("constant of TensorType must be initialized by "
                           "a DenseFPElementsAttr, got ")
             << opaqueValue;

    // If the return type of the constant is not an unranked tensor, the shape
    // must match the shape of the attribute holding the data.
    auto resultType = type.dyn_cast<mlir::RankedTensorType>();
    if (!resultType)
      return mlir::success();

    // Check that the rank of the attribute type matches the rank of the
    // constant result type.
    auto attrType = attrValue.getType().cast<mlir::TensorType>();
    if (attrType.getRank() != resultType.getRank()) {
      return op->emitOpError("return type must match the one of the attached "
                             "value attribute: ")
             << attrType.getRank() << " != " << resultType.getRank();
    }

    // Check that each of the dimensions match between the two types.
    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
      if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
        return op->emitOpError(
                   "return type shape mismatches its attribute at dimension ")
               << dim << ": " << attrType.getShape()[dim]
               << " != " << resultType.getShape()[dim];
      }
    }
    return mlir::success();
  }
  // auto resultType = type.cast<StructType>();
  // llvm::ArrayRef<mlir::Type> resultElementTypes = resultType.getElementTypes();

  // // Verify that the initializer is an Array.
  // auto attrValue = opaqueValue.dyn_cast<ArrayAttr>();
  // if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
  //   return op->emitError("constant of StructType must be initialized by an "
  //                        "ArrayAttr with the same number of elements, got ")
  //          << opaqueValue;

  // // Check that each of the elements are valid.
  // llvm::ArrayRef<mlir::Attribute> attrElementValues = attrValue.getValue();
  // for (const auto it : llvm::zip(resultElementTypes, attrElementValues))
  //   if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), op)))
  //     return mlir::failure();
  return mlir::success();
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
mlir::LogicalResult Constantop::verify() {
  return verifyConstantForType(getResult().getType(), getValue(), *this);
}


/// Infer the output shape of the ConstantOp, this is required by the shape
/// inference interface.
// void ConstantOp::inferShapes() { getResult().setType(getValue().getType()); }


//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void Addop::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult Addop::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void Addop::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }
