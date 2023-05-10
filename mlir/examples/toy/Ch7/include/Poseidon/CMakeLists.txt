set(LLVM_TARGET_DEFINITIONS PoseidonOps.td)
mlir_tablegen(PoseidonDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PoseidonDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(PoseidonOps.h.inc -gen-op-decls)
mlir_tablegen(PoseidonOps.cpp.inc -gen-op-defs)


add_mlir_dialect(PoseidonOps poseidon)
add_mlir_doc(PoseidonDialect PoseidonDialect Poseidon/ -gen-dialect-doc)
add_mlir_doc(PoseidonOps PoseidonOps Poseidon/ -gen-op-doc)

add_public_tablegen_target(ToyCh7POpsIncGen)
