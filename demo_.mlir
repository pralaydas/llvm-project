module {
  func.func @main() {
    %alloc = memref.alloc() : memref<8xf64>
    %alloc_0 = memref.alloc() : memref<8xf64>
    %alloc_1 = memref.alloc() : memref<8xf64>
    %cst = arith.constant dense<[1.200000e+00, 2.100000e+00, 4.700000e+00, 1.900000e+00, 1.100000e+00, 2.900000e+00, 2.700000e+00, 7.500000e+00]> : tensor<8xf64>
    memref.tensor_store %cst, %alloc_1 : memref<8xf64>
    %cst_2 = arith.constant dense<[1.200000e+00, 2.100000e+00, 4.700000e+00, 1.900000e+00, 1.100000e+00, 2.900000e+00, 2.700000e+00, 7.500000e+00]> : tensor<8xf64>
    memref.tensor_store %cst_2, %alloc_0 : memref<8xf64>
    affine.for %arg0 = 0 to 8 step 4 {
      %0 = affine.vector_load %alloc_1[%arg0] : memref<8xf64>, vector<4xf64>
      %1 = affine.vector_load %alloc_0[%arg0] : memref<8xf64>, vector<4xf64>
      %2 = poseidon.add %0, %1 : vector<4xf64>
      affine.vector_store %2, %alloc[%arg0] : memref<8xf64>, vector<4xf64>
    }
    memref.dealloc %alloc_1 : memref<8xf64>
    memref.dealloc %alloc_0 : memref<8xf64>
    memref.dealloc %alloc : memref<8xf64>
    return
  }
}

module {
  memref.global "private" constant @__constant : memref<8xf64> = dense<[1.200000e+00, 2.100000e+00, 4.700000e+00, 1.900000e+00, 1.100000e+00, 2.900000e+00, 2.700000e+00, 7.500000e+00]>
  func.func @main() {
    %0 = memref.get_global @__constant : memref<8xf64>
    return
  }
}