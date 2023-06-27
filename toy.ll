
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@frmt_spec = internal constant [4 x i8] c"%f \00"
@constant_1 = private unnamed_addr constant [8 x double] [double 1.100000e+00, double 2.400000e+00, double 4.700000e+00, double 1.300000e+00, double 1.000000e+00, double 2.100000e+00, double 2.300000e+00, double 7.200000e+00]
@constant_0 = private unnamed_addr constant [8 x double] [double 1.300000e+00, double 2.600000e+00, double 4.200000e+00, double 1.100000e+00, double 1.900000e+00, double 2.200000e+00, double 2.500000e+00, double 7.700000e+00]

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #0

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr nocapture noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #2

; Function Attrs: nounwind
define void @main() local_unnamed_addr #3 {
.preheader.preheader:
  %0 = tail call dereferenceable_or_null(64) ptr @malloc(i64 64)
  %1 = tail call dereferenceable_or_null(64) ptr @malloc(i64 64)
  %2 = tail call dereferenceable_or_null(64) ptr @malloc(i64 64)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) %2, ptr noundef nonnull align 16 dereferenceable(64) @constant_0, i64 64, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) %1, ptr noundef nonnull align 16 dereferenceable(64) @constant_1, i64 64, i1 false)
  %3 = load <4 x double>, ptr %2, align 8
  %4 = load <4 x double>, ptr %1, align 8
  %5 = fadd <4 x double> %3, %4
  store <4 x double> %5, ptr %0, align 8
  %6 = getelementptr double, ptr %2, i64 4
  %7 = load <4 x double>, ptr %6, align 8
  %8 = getelementptr double, ptr %1, i64 4
  %9 = load <4 x double>, ptr %8, align 8
  %10 = fadd <4 x double> %7, %9
  %11 = getelementptr double, ptr %0, i64 4
  store <4 x double> %10, ptr %11, align 8
  %12 = load double, ptr %0, align 8
  %13 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %12)
  %14 = getelementptr double, ptr %0, i64 1
  %15 = load double, ptr %14, align 8
  %16 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %15)
  %17 = getelementptr double, ptr %0, i64 2
  %18 = load double, ptr %17, align 8
  %19 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %18)
  %20 = getelementptr double, ptr %0, i64 3
  %21 = load double, ptr %20, align 8
  %22 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %21)
  %23 = getelementptr double, ptr %0, i64 4
  %24 = load double, ptr %23, align 8
  %25 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %24)
  %26 = getelementptr double, ptr %0, i64 5
  %27 = load double, ptr %26, align 8
  %28 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %27)
  %29 = getelementptr double, ptr %0, i64 6
  %30 = load double, ptr %29, align 8
  %31 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %30)
  %32 = getelementptr double, ptr %0, i64 7
  %33 = load double, ptr %32, align 8
  %34 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %33)
  tail call void @free(ptr %2)
  tail call void @free(ptr %1)
  tail call void @free(ptr %0)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #4

attributes #0 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
attributes #1 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
attributes #2 = { nofree nounwind }
attributes #3 = { nounwind }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
