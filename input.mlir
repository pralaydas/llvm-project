module {
  func.func @main() {
    %cst = arith.constant 3.800000e+00 : f64
    %cst_0 = arith.constant 2.300000e+00 : f64
    %cst_1 = arith.constant 1.700000e+00 : f64
    %cst_2 = arith.constant 6.000000e+00 : f64
    %cst_3 = arith.constant 5.000000e+00 : f64
    %cst_4 = arith.constant 4.000000e+00 : f64
    %cst_5 = arith.constant 3.600000e+00 : f64
    %cst_6 = arith.constant 2.900000e+00 : f64
    %cst_7 = arith.constant 1.100000e+00 : f64
    %cst_8 = arith.constant 3.300000e+00 : f64
    %cst_9 = arith.constant 2.100000e+00 : f64
    %cst_10 = arith.constant 1.200000e+00 : f64
    %alloc = memref.alloc() : memref<1x1x1x2x3xf64>
    %alloc_11 = memref.alloc() : memref<1x1x1x2x3xf64>
    %alloc_12 = memref.alloc() : memref<1x1x1x2x3xf64>
    affine.store %cst_10, %alloc_12[0, 0, 0, 0, 0] : memref<1x1x1x2x3xf64>
    affine.store %cst_9, %alloc_12[0, 0, 0, 0, 1] : memref<1x1x1x2x3xf64>
    affine.store %cst_8, %alloc_12[0, 0, 0, 0, 2] : memref<1x1x1x2x3xf64>
    affine.store %cst_7, %alloc_12[0, 0, 0, 1, 0] : memref<1x1x1x2x3xf64>
    affine.store %cst_6, %alloc_12[0, 0, 0, 1, 1] : memref<1x1x1x2x3xf64>
    affine.store %cst_5, %alloc_12[0, 0, 0, 1, 2] : memref<1x1x1x2x3xf64>
    affine.store %cst_4, %alloc_11[0, 0, 0, 0, 0] : memref<1x1x1x2x3xf64>
    affine.store %cst_3, %alloc_11[0, 0, 0, 0, 1] : memref<1x1x1x2x3xf64>
    affine.store %cst_2, %alloc_11[0, 0, 0, 0, 2] : memref<1x1x1x2x3xf64>
    affine.store %cst_1, %alloc_11[0, 0, 0, 1, 0] : memref<1x1x1x2x3xf64>
    affine.store %cst_0, %alloc_11[0, 0, 0, 1, 1] : memref<1x1x1x2x3xf64>
    affine.store %cst, %alloc_11[0, 0, 0, 1, 2] : memref<1x1x1x2x3xf64>
    affine.for %arg0 = 0 to 1 {
      affine.for %arg1 = 0 to 1 {
        affine.for %arg2 = 0 to 1 {
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 3 {
              %0 = affine.load %alloc_12[%arg0, %arg1, %arg2, %arg3, %arg4] : memref<1x1x1x2x3xf64>
              %1 = affine.load %alloc_11[%arg0, %arg1, %arg2, %arg3, %arg4] : memref<1x1x1x2x3xf64>
              %2 = arith.addf %0, %1 : f64
              affine.store %2, %alloc[%arg0, %arg1, %arg2, %arg3, %arg4] : memref<1x1x1x2x3xf64>
            }
          }
        }
      }
    }
    // toy.print %alloc : memref<1x1x1x2x3xf64>
    memref.dealloc %alloc_12 : memref<1x1x1x2x3xf64>
    memref.dealloc %alloc_11 : memref<1x1x1x2x3xf64>
    memref.dealloc %alloc : memref<1x1x1x2x3xf64>
    return
  }
}
