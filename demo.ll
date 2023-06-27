; ModuleID = 'demo.cpp'
source_filename = "demo.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef <8 x float> @_Z3foov() #0 {
entry:
  %a = alloca <8 x float>, align 32
  %b = alloca <8 x float>, align 32
  %c = alloca <8 x float>, align 32
  store <8 x float> <float 1.500000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00, float 8.000000e+00>, ptr %a, align 32
  store <8 x float> <float 2.500000e+00, float 1.000000e+00, float 3.000000e+00, float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00, float 8.000000e+00>, ptr %b, align 32
  %0 = load <8 x float>, ptr %a, align 32
  %1 = load <8 x float>, ptr %b, align 32
  %add = fadd <8 x float> %0, %1
  store <8 x float> %add, ptr %c, align 32
  %2 = load <8 x float>, ptr %c, align 32
  ret <8 x float> %2
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="256" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 16.0.4 (https://github.com/pralaydas/llvm-project.git 17c0f8495de332006e966bc4fea68cdcc327c070)"}
