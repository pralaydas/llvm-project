; ModuleID = 'toy.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@frmt_spec = internal constant [4 x i8] c"%f \00"
@constant_1 = private unnamed_addr constant [16 x float] [float 1.000000e+00, float 2.500000e+00, float 4.000000e+00, float 1.000000e+00, float 1.000000e+00, float 2.000000e+00, float 2.000000e+00, float 7.000000e+00, float 1.500000e+00, float 2.000000e+00, float 4.000000e+00, float 1.000000e+00, float 1.000000e+00, float 2.000000e+00, float 2.000000e+00, float 7.500000e+00]
@constant_0 = private unnamed_addr constant [16 x float] [float 1.000000e+00, float 2.000000e+00, float 4.000000e+00, float 1.500000e+00, float 1.000000e+00, float 2.000000e+00, float 2.000000e+00, float 7.500000e+00, float 1.000000e+00, float 2.000000e+00, float 4.000000e+00, float 1.000000e+00, float 1.000000e+00, float 2.000000e+00, float 2.500000e+00, float 7.000000e+00]

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
  %3 = load <16 x float>, ptr %2, align 8
  %4 = load <16 x float>, ptr %1, align 8
  %5 = fadd <16 x float> %3, %4
  store <16 x float> %5, ptr %0, align 8
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
