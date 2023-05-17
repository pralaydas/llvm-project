//===- PoseidonOps.td - Poseidon dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_OPS_TD
#define POSEIDON_OPS_TD

include "Poseidon/PoseidonDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"



//===----------------------------------------------------------------------===//
// Poseidon Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def Constantop : Poseidon_Op<"constant", [ConstantLike, Pure]> {
    // Provide a summary and description for this operation. This can be used to
    // auto-generate documentation of the operations within our dialect.
    let summary = "constant";
    let description = [{
        The `poseidon.constant` operation illustrates how to define a constant
        operation in a dialect. Constant operation turns a literal into an 
        SSA value. The data is attached to the operation as an attribute.

        Example:

        ```mlir
        %0 = poseidon.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                                : tensor<2x3xf64>
        ```
    }];

    // The constant operation takes an attribute as the only input.
    let arguments = (ins F64ElementsAttr:$value);

    // The constant operation returns a single value of TensorType.
    let results = (outs F64Tensor);

    // Indicate that the operation has a custom parser and printer method.
    let hasCustomAssemblyFormat = 1;

    // Add custom build methods for the constant operation. These method populates
    // the `state` that MLIR uses to create operations, i.e. these are used when
    // using `builder.create<ConstantOp>(...)`.
    let builders = [
        // Build a constant with a given constant tensor value.
        OpBuilder<(ins "DenseElementsAttr":$value), [{
        build($_builder, $_state, value.getType(), value);
        }]>,

        // Build a constant with a given constant floating-point value.
        // OpBuilder<(ins "double":$value)>
    ];

    // Indicate that additional verification for this operation is necessary.
    let hasVerifier = 1;

    // Set the folder bit so that we can implement constant folders.
    // let hasFolder = 1;
    // let extraClassDeclaration = [{
    // /// Whether the constant op can be constructed with a particular value and
    // /// type.
    // static bool isBuildableWith(Attribute value, Type type);

    // /// Build the constant op with `value` and `type` if possible, otherwise
    // /// returns null.
    // static Constantop materialize(OpBuilder &builder, Attribute value,
    //                               Type type, Location loc);
    // }];
}


//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def Addop : Poseidon_Op<"add", [Pure, Commutative]>{
    let summary = "element-wise addition operation";
    let description = [{
        The 'poseidon.add' operation performs element-wise addition between two tensors.
        The shapes of the tensor operands are expected to match.
        Example:

        ```mlir
        %2 = poseidon.add (%0, %1) : (tensor<6xf64>, tensor<6xf64>) -> tensor<*xf64>
        ```
    }];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    // Indicate that the operation has a custom parser and printer method.
    let hasCustomAssemblyFormat = 1;

    // Allow building an AddOp with from the two input operands.
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
    

}


#endif // POSEIDON_OPS_TD
