//===- Passes.h - Toy Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Toy.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PASSES_H
#define POSEIDON_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace poseidon {


/// Create a pass for lowering to operations in the `Poseidon` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToPoseidonPass();
std::unique_ptr<mlir::Pass> createLowerToPoseidonLoopsPass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
std::unique_ptr<mlir::Pass> createPrintPass();

} // namespace poseidon
} // namespace mlir

#endif // POSEIDON_PASSES_H
