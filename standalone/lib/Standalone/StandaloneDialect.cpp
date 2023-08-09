//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"


#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"



using namespace mlir;
using namespace mlir::standalone;

#include "Standalone/StandaloneOpsDialect.cpp.inc"


//===----------------------------------------------------------------------===//
// StandaloneDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct StandaloneInlinerInterface : public DialectInlinerInterface {
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
// Standalone dialect.
//===----------------------------------------------------------------------===//

void StandaloneDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/StandaloneOps.cpp.inc"
      >();
  addInterfaces<StandaloneInlinerInterface>();
}

