//===- PoseidonDialect.td - Poseidon dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_DIALECT
#define POSEIDON_DIALECT

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
}

//===----------------------------------------------------------------------===//
// Base poseidon operation definition.
//===----------------------------------------------------------------------===//

class Poseidon_Op<string mnemonic, list<Trait> traits = []> :
        Op<Poseidon_Dialect, mnemonic, traits>;

#endif // POSEIDON_DIALECT
