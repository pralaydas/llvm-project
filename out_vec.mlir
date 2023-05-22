module {
  func.func @main() {
    %cst = arith.constant 6.000000e+00 : f64
    %cst_0 = arith.constant 7.000000e+00 : f64
    %cst_1 = arith.constant 5.000000e+00 : f64
    %cst_2 = arith.constant 4.000000e+00 : f64
    %cst_3 = arith.constant 3.000000e+00 : f64
    %cst_4 = arith.constant 2.000000e+00 : f64
    %cst_5 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<2x8x8xf64>
    %alloc_6 = memref.alloc() : memref<2x8x8xf64>
    %alloc_7 = memref.alloc() : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 0, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 0, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 0, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 0, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 0, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 0, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 0, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 0, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_7[0, 1, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 1, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 1, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 1, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 1, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 1, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 1, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 1, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 2, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 2, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 2, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 2, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 2, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 2, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 2, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 2, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_7[0, 3, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 3, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 3, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 3, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 3, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 3, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 3, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 3, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 4, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 4, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 4, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 4, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 4, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 4, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 4, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 4, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_7[0, 5, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 5, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 5, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 5, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 5, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 5, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 5, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 5, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 6, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 6, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 6, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 6, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 6, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 6, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 6, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 6, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_7[0, 7, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 7, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 7, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 7, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[0, 7, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[0, 7, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[0, 7, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[0, 7, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 0, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 0, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 0, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 0, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 0, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 0, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 0, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 0, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_7[1, 1, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 1, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 1, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 1, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 1, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 1, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 1, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 1, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 2, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 2, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 2, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 2, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 2, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 2, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 2, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 2, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_7[1, 3, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 3, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 3, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 3, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 3, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 3, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 3, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 3, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 4, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 4, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 4, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 4, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 4, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 4, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 4, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 4, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_7[1, 5, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 5, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 5, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 5, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 5, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 5, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 5, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 5, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 6, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 6, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 6, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 6, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 6, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 6, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 6, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 6, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_7[1, 7, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 7, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 7, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 7, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_7[1, 7, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_7[1, 7, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_7[1, 7, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_7[1, 7, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[0, 0, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 0, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 0, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 0, 3] : memref<2x8x8xf64>
    affine.store %cst_0, %alloc_6[0, 0, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 0, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 0, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 0, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_6[0, 1, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 1, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 1, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 1, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[0, 1, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 1, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 1, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 1, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[0, 2, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 2, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 2, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 2, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[0, 2, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 2, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 2, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 2, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_6[0, 3, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 3, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 3, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 3, 3] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 3, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 3, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 3, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 3, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[0, 4, 0] : memref<2x8x8xf64>
    affine.store %cst, %alloc_6[0, 4, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 4, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 4, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[0, 4, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 4, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 4, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 4, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_6[0, 5, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 5, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 5, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 5, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[0, 5, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 5, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 5, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 5, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[0, 6, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 6, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 6, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 6, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[0, 6, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 6, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 6, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 6, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_6[0, 7, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 7, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 7, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 7, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[0, 7, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[0, 7, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[0, 7, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[0, 7, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 0, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 0, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 0, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 0, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 0, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 0, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 0, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 0, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_6[1, 1, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 1, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 1, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 1, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 1, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 1, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 1, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 1, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 2, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 2, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 2, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 2, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 2, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 2, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 2, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 2, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_6[1, 3, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 3, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 3, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 3, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 3, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 3, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 3, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 3, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 4, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 4, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 4, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 4, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 4, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 4, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 4, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 4, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_6[1, 5, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 5, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 5, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 5, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 5, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 5, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 5, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 5, 7] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 6, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 6, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 6, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 6, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 6, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 6, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 6, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 6, 7] : memref<2x8x8xf64>
    affine.store %cst_1, %alloc_6[1, 7, 0] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 7, 1] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 7, 2] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 7, 3] : memref<2x8x8xf64>
    affine.store %cst_5, %alloc_6[1, 7, 4] : memref<2x8x8xf64>
    affine.store %cst_4, %alloc_6[1, 7, 5] : memref<2x8x8xf64>
    affine.store %cst_3, %alloc_6[1, 7, 6] : memref<2x8x8xf64>
    affine.store %cst_2, %alloc_6[1, 7, 7] : memref<2x8x8xf64>
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 8 {
        affine.for %arg2 = 0 to 8 step 4 {
          %cst_8 = arith.constant 0.000000e+00 : f64
          %0 = vector.transfer_read %alloc_7[%arg0, %arg1, %arg2], %cst_8 : memref<2x8x8xf64>, vector<4xf64>
          %cst_9 = arith.constant 0.000000e+00 : f64
          %1 = vector.transfer_read %alloc_6[%arg0, %arg1, %arg2], %cst_9 : memref<2x8x8xf64>, vector<4xf64>
          %2 = arith.addf %0, %1 : vector<4xf64>
          vector.transfer_write %2, %alloc[%arg0, %arg1, %arg2] : vector<4xf64>, memref<2x8x8xf64>
        }
      }
    }
    memref.dealloc %alloc_7 : memref<2x8x8xf64>
    memref.dealloc %alloc_6 : memref<2x8x8xf64>
    memref.dealloc %alloc : memref<2x8x8xf64>
    return
  }
}

