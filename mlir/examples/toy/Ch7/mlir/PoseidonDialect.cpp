//===- PoseidonDialect.cpp - Poseidon Dialect Definition-------------------*- C++ -*-===//
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
// This file defines poseidon dialect.
//
//===----------------------------------------------------------------------===//

#include "Poseidon/PoseidonDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "Poseidon/PoseidonDialect.h"
#include "Poseidon/PoseidonOps.h"

using namespace mlir;
using namespace mlir::poseidon;

#include "Poseidon/PoseidonDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PoseidonDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct PoseidonInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }
  bool isLegalToInline(Region *, Region *, bool ,
                       IRMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Poseidon/PoseidonOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Poseidon dialect.
//===----------------------------------------------------------------------===//

void PoseidonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Poseidon/PoseidonOps.cpp.inc"
      >();
  addInterfaces<PoseidonInlinerInterface>();
}

mlir::Operation *PoseidonDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  return builder.create<Constantop>(loc, type,
                                    value.cast<mlir::DenseElementsAttr>());
}