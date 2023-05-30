//===- PoseidonDialect.td - Poseidon dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_DIALECT_TD
#define POSEIDON_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Poseidon dialect definition.
//===----------------------------------------------------------------------===//

def Poseidon_Dialect : Dialect {
    let name = "poseidon";
    let summary = "A poseidon out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::poseidon";
    let useFoldAPI = kEmitFoldAdaptorFolder;
    // let useDefaultTypePrinterParser = 1;
    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base poseidon operation definition.
//===----------------------------------------------------------------------===//

// Base class for poseidon dialect operations. This operation inherits from 
// the base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Poseidon_Op<string mnemonic, list<Trait> traits = []> :
        Op<Poseidon_Dialect, mnemonic, traits >;

#endif // POSEIDON_DIALECT_TD
