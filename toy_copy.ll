; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@nl = internal constant [2 x i8] c"\0A\00"
@frmt_spec = internal constant [4 x i8] c"%f \00"
@constant_1 = private constant [32 x double] [double 1.000000e+00, double 2.500000e+00, double 4.000000e+00, double 1.000000e+00, double 1.000000e+00, double 2.000000e+00, double 2.000000e+00, double 7.000000e+00, double 1.500000e+00, double 2.000000e+00, double 4.000000e+00, double 1.000000e+00, double 1.000000e+00, double 2.000000e+00, double 2.000000e+00, double 7.500000e+00, double 1.000000e+00, double 2.000000e+00, double 4.000000e+00, double 1.500000e+00, double 1.000000e+00, double 2.000000e+00, double 2.000000e+00, double 7.500000e+00, double 1.000000e+00, double 2.500000e+00, double 4.000000e+00, double 1.000000e+00, double 1.500000e+00, double 2.000000e+00, double 2.500000e+00, double 7.000000e+00]
@constant_0 = private constant [32 x double] [double 1.000000e+00, double 2.000000e+00, double 4.000000e+00, double 1.500000e+00, double 1.000000e+00, double 2.000000e+00, double 2.000000e+00, double 7.500000e+00, double 1.000000e+00, double 2.000000e+00, double 4.000000e+00, double 1.000000e+00, double 1.000000e+00, double 2.000000e+00, double 2.500000e+00, double 7.000000e+00, double 1.000000e+00, double 2.000000e+00, double 4.000000e+00, double 1.500000e+00, double 1.000000e+00, double 2.000000e+00, double 2.000000e+00, double 7.500000e+00, double 1.000000e+00, double 2.000000e+00, double 4.000000e+00, double 1.000000e+00, double 1.000000e+00, double 2.000000e+00, double 2.500000e+00, double 7.000000e+00]

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @printf(ptr, ...)

define void @main() {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 32) to i64))
  %2 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, i64 32, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, i64 1, 4, 0
  %7 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 32) to i64))
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %7, 0
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, ptr %7, 1
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 0, 2
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 32, 3, 0
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 1, 4, 0
  %13 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 32) to i64))
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %13, 0
  %15 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, ptr %13, 1
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, i64 0, 2
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, i64 32, 3, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 1, 4, 0
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, 1
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, 2
  %21 = getelementptr double, ptr %19, i64 %20
  call void @llvm.memcpy.p0.p0.i64(ptr %21, ptr @constant_0, i64 mul (i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), i64 32), i1 false)
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 2
  %24 = getelementptr double, ptr %22, i64 %23
  call void @llvm.memcpy.p0.p0.i64(ptr %24, ptr @constant_1, i64 mul (i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), i64 32), i1 false)
  br label %25

25:                                               ; preds = %28, %0
  %26 = phi i64 [ 0, %0 ], [ %38, %28 ]
  %27 = icmp slt i64 %26, 32
  br i1 %27, label %28, label %39

28:                                               ; preds = %25
  %29 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, 1
  %30 = getelementptr double, ptr %29, i64 %26
  %31 = load <16 x double>, ptr %30, align 8
  %32 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %33 = getelementptr double, ptr %32, i64 %26
  %34 = load <16 x double>, ptr %33, align 8
  %35 = fadd <16 x double> %31, %34
  %36 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %37 = getelementptr double, ptr %36, i64 %26
  store <16 x double> %35, ptr %37, align 8
  %38 = add i64 %26, 16
  br label %25

39:                                               ; preds = %25
  br label %40

40:                                               ; preds = %43, %39
  %41 = phi i64 [ 0, %39 ], [ %48, %43 ]
  %42 = icmp slt i64 %41, 32
  br i1 %42, label %43, label %49

43:                                               ; preds = %40
  %44 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %45 = getelementptr double, ptr %44, i64 %41
  %46 = load double, ptr %45, align 8
  %47 = call i32 (ptr, ...) @printf(ptr @frmt_spec, double %46)
  %48 = add i64 %41, 1
  br label %40

49:                                               ; preds = %40
  %50 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, 0
  call void @free(ptr %50)
  %51 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 0
  call void @free(ptr %51)
  %52 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  call void @free(ptr %52)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}