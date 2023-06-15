func.func @basic() -> tensor<4xf32> {
  %0 = arith.constant dense<7.0> : tensor<4xf32>
  %1 = arith.constant dense<6.0> : tensor<4xf32>
  %3 = arith.addf %0, %1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<3x4xf32>
}