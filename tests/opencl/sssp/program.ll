; ModuleID = '/home/ecegridfs/a/socet177/.cache/pocl/uncached/tempfile_cTGfDf.cl'
source_filename = "/home/ecegridfs/a/socet177/.cache/pocl/uncached/tempfile_cTGfDf.cl"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown-elf"

%union.dim3_t = type { %struct.anon }
%struct.anon = type { i32, i32, i32 }

@blockDim = external local_unnamed_addr global %union.dim3_t, align 4
@g_global_offset = external local_unnamed_addr global %union.dim3_t, align 4
@blockIdx = external thread_local global %union.dim3_t, align 4
@threadIdx = external thread_local global %union.dim3_t, align 4

define void @OCL_SSSP_KERNEL1(ptr %ArgBuffer) {
entry:
  %vertexArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 0
  %vertexArray_loaded = load ptr, ptr %vertexArray_offset_ptr, align 4, !vortex.uniform !32
  %edgeArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 4
  %edgeArray_loaded = load ptr, ptr %edgeArray_offset_ptr, align 4, !vortex.uniform !32
  %weightArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 8
  %weightArray_loaded = load ptr, ptr %weightArray_offset_ptr, align 4, !vortex.uniform !32
  %maskArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 12
  %maskArray_loaded = load ptr, ptr %maskArray_offset_ptr, align 4, !vortex.uniform !32
  %costArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 16
  %costArray_loaded = load ptr, ptr %costArray_offset_ptr, align 4, !vortex.uniform !32
  %updatingCostArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 20
  %updatingCostArray_loaded = load ptr, ptr %updatingCostArray_offset_ptr, align 4, !vortex.uniform !32
  %vertexCount_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 24
  %vertexCount_loaded = load i32, ptr %vertexCount_offset_ptr, align 4, !vortex.uniform !32
  %edgeCount_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 28
  %edgeCount_loaded = load i32, ptr %edgeCount_offset_ptr, align 4, !vortex.uniform !32
  br label %entry1

entry1:                                           ; preds = %entry
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @blockIdx)
  %1 = load i32, ptr %0, align 4, !tbaa !33
  %2 = load i32, ptr @blockDim, align 4, !tbaa !33
  %3 = mul i32 %2, %1
  %4 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @threadIdx)
  %5 = load i32, ptr %4, align 4, !tbaa !33
  %6 = add i32 %5, %3
  %7 = load i32, ptr @g_global_offset, align 4, !tbaa !33
  %8 = add i32 %6, %7
  %arrayidx = getelementptr inbounds i32, ptr %maskArray_loaded, i32 %8
  %9 = load i32, ptr %arrayidx, align 4, !tbaa !36
  %cmp.not = icmp eq i32 %9, 0
  br i1 %cmp.not, label %if.end20, label %if.then

if.then:                                          ; preds = %entry1
  store i32 0, ptr %arrayidx, align 4, !tbaa !36
  %arrayidx2 = getelementptr inbounds i32, ptr %vertexArray_loaded, i32 %8
  %10 = load i32, ptr %arrayidx2, align 4, !tbaa !36
  %add = add nsw i32 %8, 1
  %cmp3 = icmp slt i32 %add, %vertexCount_loaded
  br i1 %cmp3, label %if.then4, label %if.end

if.then4:                                         ; preds = %if.then
  %arrayidx6 = getelementptr inbounds i32, ptr %vertexArray_loaded, i32 %add
  %11 = load i32, ptr %arrayidx6, align 4, !tbaa !36
  br label %if.end

if.end:                                           ; preds = %if.then, %if.then4
  %edgeEnd.0 = phi i32 [ %11, %if.then4 ], [ %edgeCount_loaded, %if.then ]
  %cmp737 = icmp slt i32 %10, %edgeEnd.0
  br i1 %cmp737, label %for.body.lr.ph, label %if.end20

for.body.lr.ph:                                   ; preds = %if.end
  %arrayidx10 = getelementptr inbounds float, ptr %costArray_loaded, i32 %8
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %if.end19
  %edge.038 = phi i32 [ %10, %for.body.lr.ph ], [ %inc, %if.end19 ]
  %arrayidx8 = getelementptr inbounds i32, ptr %edgeArray_loaded, i32 %edge.038
  %12 = load i32, ptr %arrayidx8, align 4, !tbaa !36
  %arrayidx9 = getelementptr inbounds float, ptr %updatingCostArray_loaded, i32 %12
  %13 = load float, ptr %arrayidx9, align 4, !tbaa !38
  %14 = load float, ptr %arrayidx10, align 4, !tbaa !38
  %arrayidx11 = getelementptr inbounds float, ptr %weightArray_loaded, i32 %edge.038
  %15 = load float, ptr %arrayidx11, align 4, !tbaa !38
  %add12 = fadd float %14, %15
  %cmp13 = fcmp ogt float %13, %add12
  br i1 %cmp13, label %if.then14, label %if.end19

if.then14:                                        ; preds = %for.body
  store float %add12, ptr %arrayidx9, align 4, !tbaa !38
  br label %if.end19

if.end19:                                         ; preds = %if.then14, %for.body
  %inc = add nsw i32 %edge.038, 1
  %exitcond.not = icmp eq i32 %inc, %edgeEnd.0
  br i1 %exitcond.not, label %if.end20, label %for.body

if.end20:                                         ; preds = %if.end19, %if.end, %entry1
  ret void
}

define void @OCL_SSSP_KERNEL2(ptr %ArgBuffer) {
entry:
  %vertexArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 0
  %vertexArray_loaded = load ptr, ptr %vertexArray_offset_ptr, align 4, !vortex.uniform !32
  %edgeArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 4
  %edgeArray_loaded = load ptr, ptr %edgeArray_offset_ptr, align 4, !vortex.uniform !32
  %weightArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 8
  %weightArray_loaded = load ptr, ptr %weightArray_offset_ptr, align 4, !vortex.uniform !32
  %maskArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 12
  %maskArray_loaded = load ptr, ptr %maskArray_offset_ptr, align 4, !vortex.uniform !32
  %costArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 16
  %costArray_loaded = load ptr, ptr %costArray_offset_ptr, align 4, !vortex.uniform !32
  %updatingCostArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 20
  %updatingCostArray_loaded = load ptr, ptr %updatingCostArray_offset_ptr, align 4, !vortex.uniform !32
  %vertexCount_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 24
  %vertexCount_loaded = load i32, ptr %vertexCount_offset_ptr, align 4, !vortex.uniform !32
  br label %entry1

entry1:                                           ; preds = %entry
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @blockIdx)
  %1 = load i32, ptr %0, align 4, !tbaa !33
  %2 = load i32, ptr @blockDim, align 4, !tbaa !33
  %3 = mul i32 %2, %1
  %4 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @threadIdx)
  %5 = load i32, ptr %4, align 4, !tbaa !33
  %6 = add i32 %5, %3
  %7 = load i32, ptr @g_global_offset, align 4, !tbaa !33
  %8 = add i32 %6, %7
  %arrayidx = getelementptr inbounds float, ptr %costArray_loaded, i32 %8
  %9 = load float, ptr %arrayidx, align 4, !tbaa !38
  %arrayidx1 = getelementptr inbounds float, ptr %updatingCostArray_loaded, i32 %8
  %10 = load float, ptr %arrayidx1, align 4, !tbaa !38
  %cmp = fcmp ogt float %9, %10
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry1
  store float %10, ptr %arrayidx, align 4, !tbaa !38
  %arrayidx4 = getelementptr inbounds i32, ptr %maskArray_loaded, i32 %8
  store i32 1, ptr %arrayidx4, align 4, !tbaa !36
  br label %if.end

if.end:                                           ; preds = %if.then, %entry1
  %11 = phi float [ %10, %if.then ], [ %9, %entry1 ]
  store float %11, ptr %arrayidx1, align 4, !tbaa !38
  ret void
}

define void @initializeBuffers(ptr %ArgBuffer) {
entry:
  %maskArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 0
  %maskArray_loaded = load ptr, ptr %maskArray_offset_ptr, align 4, !vortex.uniform !32
  %costArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 4
  %costArray_loaded = load ptr, ptr %costArray_offset_ptr, align 4, !vortex.uniform !32
  %updatingCostArray_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 8
  %updatingCostArray_loaded = load ptr, ptr %updatingCostArray_offset_ptr, align 4, !vortex.uniform !32
  %sourceVertex_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 12
  %sourceVertex_loaded = load i32, ptr %sourceVertex_offset_ptr, align 4, !vortex.uniform !32
  %vertexCount_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 16
  %vertexCount_loaded = load i32, ptr %vertexCount_offset_ptr, align 4, !vortex.uniform !32
  br label %entry1

entry1:                                           ; preds = %entry
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @blockIdx)
  %1 = load i32, ptr %0, align 4, !tbaa !33
  %2 = load i32, ptr @blockDim, align 4, !tbaa !33
  %3 = mul i32 %2, %1
  %4 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @threadIdx)
  %5 = load i32, ptr %4, align 4, !tbaa !33
  %6 = add i32 %5, %3
  %7 = load i32, ptr @g_global_offset, align 4, !tbaa !33
  %8 = add i32 %6, %7
  %cmp = icmp eq i32 %8, %sourceVertex_loaded
  %spec.select = zext i1 %cmp to i32
  %spec.select17 = select i1 %cmp, float 0.000000e+00, float 0x47EFFFFFE0000000
  %9 = getelementptr inbounds i32, ptr %maskArray_loaded, i32 %8
  store i32 %spec.select, ptr %9, align 4
  %10 = getelementptr inbounds float, ptr %costArray_loaded, i32 %8
  store float %spec.select17, ptr %10, align 4
  %11 = getelementptr inbounds float, ptr %updatingCostArray_loaded, i32 %8
  store float %spec.select17, ptr %11, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull) #0

define ptr @__vx_get_kernel_callback(i32 %kernel_index) {
entry:
  switch i32 %kernel_index, label %default [
    i32 0, label %case_0
    i32 1, label %case_1
    i32 2, label %case_2
  ]

case_0:                                           ; preds = %entry
  ret ptr @OCL_SSSP_KERNEL1

case_1:                                           ; preds = %entry
  ret ptr @OCL_SSSP_KERNEL2

case_2:                                           ; preds = %entry
  ret ptr @initializeBuffers

default:                                          ; preds = %entry
  ret ptr null
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !4, !5}
!opencl.ocl.version = !{!6}
!llvm.ident = !{!7}
!pocl_meta = !{!8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"ilp32f"}
!2 = !{i32 6, !"riscv-isa", !3}
!3 = !{!"rv32i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0"}
!4 = !{i32 8, !"SmallDataLimit", i32 0}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 1, i32 2}
!7 = !{!"clang version 18.1.7 (https://github.com/vortexgpgpu/llvm.git b115a172abc24683b2730b5b601f34e27fe19d93)"}
!8 = !{!"device_aux_functions", !""}
!9 = !{!"device_address_bits", i64 32}
!10 = !{!"device_arg_buffer_launcher", i8 0}
!11 = !{!"device_grid_launcher", i8 0}
!12 = !{!"device_is_spmd", i8 1}
!13 = !{!"device_native_vec_width", i64 256}
!14 = !{!"WGMaxGridDimWidth", i64 0}
!15 = !{!"WGLocalSizeX", i64 0}
!16 = !{!"WGLocalSizeY", i64 0}
!17 = !{!"WGLocalSizeZ", i64 0}
!18 = !{!"WGDynamicLocalSize", i8 1}
!19 = !{!"WGAssumeZeroGlobalOffset", i8 0}
!20 = !{!"device_global_as_id", i64 0}
!21 = !{!"device_local_as_id", i64 0}
!22 = !{!"device_constant_as_id", i64 0}
!23 = !{!"device_args_as_id", i64 0}
!24 = !{!"device_context_as_id", i64 0}
!25 = !{!"device_side_printf", i8 0}
!26 = !{!"device_alloca_locals", i8 0}
!27 = !{!"device_autolocals_to_args", i64 1}
!28 = !{!"device_max_witem_dim", i64 3}
!29 = !{!"device_max_witem_sizes_0", i64 8}
!30 = !{!"device_max_witem_sizes_1", i64 8}
!31 = !{!"device_max_witem_sizes_2", i64 8}
!32 = !{!"vortex.uniform"}
!33 = !{!34, !34, i64 0}
!34 = !{!"omnipotent char", !35, i64 0}
!35 = !{!"Simple C/C++ TBAA"}
!36 = !{!37, !37, i64 0}
!37 = !{!"int", !34, i64 0}
!38 = !{!39, !39, i64 0}
!39 = !{!"float", !34, i64 0}
