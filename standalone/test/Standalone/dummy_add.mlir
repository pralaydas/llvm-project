// RUN: standalone-opt %s | standalone-opt | FileCheck %s


// CHECK-LABEL: func.func @bar(%{{.*}}: f32, %{{.*}}: f32) -> f32
func.func @bar(%arg0: f32, %arg1: f32) -> f32{
    // CHECK-NEXT: %{{.*}} = arith.constant 1.500000e+00 : f32 
    // CHECK-NEXT: %{{.*}} = arith.constant 2.500000e+00 : f32 
    // CHECK-NEXT: %{{.*}} = standalone.add %{{.*}}, %{{.*}} : f32
    // CHECK-NEXT: %{{.*}} = standalone.add %{{.*}}, %{{.*}} : f32
    // CHECK-NEXT: %{{.*}} = standalone.add %{{.*}}, %{{.*}} : f32 
    // CHECK-NEXT: return %{{.*}} : f32
    // CHECK-NEXT: }
    %0 = arith.constant 1.500000e+00 : f32
    %1 = arith.constant 2.500000e+00 : f32
    %2 = standalone.add %1, %0 : f32
    %3 = standalone.add %arg0, %arg1 : f32
    %4 = standalone.add %2, %3 : f32
    return %4 : f32
}

