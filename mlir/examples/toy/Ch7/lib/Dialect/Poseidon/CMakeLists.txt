add_mlir_dialect_library(Poseidon
  PoseidonDialect.cpp
  PoseidonOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  )
