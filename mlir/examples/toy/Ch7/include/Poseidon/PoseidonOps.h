//===- PoseidonOps.h - Poseidon dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_POSEIDONOPS_H
#define POSEIDON_POSEIDONOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Poseidon Operations
//===----------------------------------------------------------------------===//

/// Include the auto-generated header file containing the declarations of the
/// poseidon operations.
#define GET_OP_CLASSES
#include "Poseidon/PoseidonOps.h.inc"



#endif // POSEIDON_POSEIDONOPS_H
