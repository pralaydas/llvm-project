//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "Standalone/StandaloneDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


//===----------------------------------------------------------------------===//
// FooOp
//===----------------------------------------------------------------------===//

def Standalone_FooOp : Standalone_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = standalone.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}


//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//
def AddOp : Standalone_Op<"add", [Pure, Commutative]>{
    let summary = "element-wise addition operation";
    let description = [{
        The 'standalone.add' operation performs element-wise addition between two tensors.
        The shapes of the tensor operands are expected to match.
        Example:

        ```mlir
        %2 = standalone.add (%0, %1) : (tensor<6xf64>, tensor<6xf64>) -> tensor<*xf64>
        ```
    }];
    let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
    let results = (outs AnyFloat:$result);

    // Indicate that the operation has a custom parser and printer method.
    let hasCustomAssemblyFormat = 1;

    // Allow building an AddOp with from the two input operands.
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
    

}




#endif // STANDALONE_OPS
