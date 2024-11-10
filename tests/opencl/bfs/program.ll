; ModuleID = '/home/ecegridfs/a/socet177/.cache/pocl/uncached/tempfile_rzrI2G.cl'
source_filename = "/home/ecegridfs/a/socet177/.cache/pocl/uncached/tempfile_rzrI2G.cl"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown-elf"

%union.dim3_t = type { %struct.anon }
%struct.anon = type { i32, i32, i32 }
%struct.Node = type { i32, i32 }

@blockDim = external local_unnamed_addr global %union.dim3_t, align 4
@g_global_offset = external local_unnamed_addr global %union.dim3_t, align 4
@blockIdx = external thread_local global %union.dim3_t, align 4
@threadIdx = external thread_local global %union.dim3_t, align 4

define void @BFS_1(ptr %ArgBuffer) {
entry:
  %g_graph_nodes_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 0
  %g_graph_nodes_loaded = load ptr, ptr %g_graph_nodes_offset_ptr, align 4, !vortex.uniform !32
  %g_graph_edges_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 4
  %g_graph_edges_loaded = load ptr, ptr %g_graph_edges_offset_ptr, align 4, !vortex.uniform !32
  %g_graph_mask_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 8
  %g_graph_mask_loaded = load ptr, ptr %g_graph_mask_offset_ptr, align 4, !vortex.uniform !32
  %g_updating_graph_mask_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 12
  %g_updating_graph_mask_loaded = load ptr, ptr %g_updating_graph_mask_offset_ptr, align 4, !vortex.uniform !32
  %g_graph_visited_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 16
  %g_graph_visited_loaded = load ptr, ptr %g_graph_visited_offset_ptr, align 4, !vortex.uniform !32
  %g_cost_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 20
  %g_cost_loaded = load ptr, ptr %g_cost_offset_ptr, align 4, !vortex.uniform !32
  %no_of_nodes_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 24
  %no_of_nodes_loaded = load i32, ptr %no_of_nodes_offset_ptr, align 4, !vortex.uniform !32
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
  %cmp = icmp slt i32 %8, %no_of_nodes_loaded
  br i1 %cmp, label %land.lhs.true, label %if.end16

land.lhs.true:                                    ; preds = %entry1
  %arrayidx = getelementptr inbounds i8, ptr %g_graph_mask_loaded, i32 %8
  %9 = load i8, ptr %arrayidx, align 1, !tbaa !33
  %tobool.not = icmp eq i8 %9, 0
  br i1 %tobool.not, label %if.end16, label %if.then

if.then:                                          ; preds = %land.lhs.true
  store i8 0, ptr %arrayidx, align 1, !tbaa !33
  %arrayidx2 = getelementptr inbounds %struct.Node, ptr %g_graph_nodes_loaded, i32 %8
  %no_of_edges = getelementptr inbounds %struct.Node, ptr %g_graph_nodes_loaded, i32 %8, i32 1
  %10 = load i32, ptr %no_of_edges, align 4, !tbaa !36
  %cmp632 = icmp sgt i32 %10, 0
  br i1 %cmp632, label %for.body.lr.ph, label %if.end16

for.body.lr.ph:                                   ; preds = %if.then
  %11 = load i32, ptr %arrayidx2, align 4, !tbaa !39
  %arrayidx12 = getelementptr inbounds i32, ptr %g_cost_loaded, i32 %8
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %if.end
  %12 = phi i32 [ %11, %for.body.lr.ph ], [ %17, %if.end ]
  %13 = phi i32 [ %10, %for.body.lr.ph ], [ %18, %if.end ]
  %i.033 = phi i32 [ %11, %for.body.lr.ph ], [ %inc, %if.end ]
  %arrayidx8 = getelementptr inbounds i32, ptr %g_graph_edges_loaded, i32 %i.033
  %14 = load i32, ptr %arrayidx8, align 4, !tbaa !40
  %arrayidx9 = getelementptr inbounds i8, ptr %g_graph_visited_loaded, i32 %14
  %15 = load i8, ptr %arrayidx9, align 1, !tbaa !33
  %tobool10.not = icmp eq i8 %15, 0
  br i1 %tobool10.not, label %if.then11, label %if.end

if.then11:                                        ; preds = %for.body
  %16 = load i32, ptr %arrayidx12, align 4, !tbaa !40
  %add13 = add nsw i32 %16, 1
  %arrayidx14 = getelementptr inbounds i32, ptr %g_cost_loaded, i32 %14
  store i32 %add13, ptr %arrayidx14, align 4, !tbaa !40
  %arrayidx15 = getelementptr inbounds i8, ptr %g_updating_graph_mask_loaded, i32 %14
  store i8 1, ptr %arrayidx15, align 1, !tbaa !33
  %.pre = load i32, ptr %no_of_edges, align 4, !tbaa !36
  %.pre34 = load i32, ptr %arrayidx2, align 4, !tbaa !39
  br label %if.end

if.end:                                           ; preds = %if.then11, %for.body
  %17 = phi i32 [ %.pre34, %if.then11 ], [ %12, %for.body ]
  %18 = phi i32 [ %.pre, %if.then11 ], [ %13, %for.body ]
  %inc = add nsw i32 %i.033, 1
  %add = add nsw i32 %18, %17
  %cmp6 = icmp slt i32 %inc, %add
  br i1 %cmp6, label %for.body, label %if.end16

if.end16:                                         ; preds = %if.end, %if.then, %land.lhs.true, %entry1
  ret void
}

define void @BFS_2(ptr %ArgBuffer) {
entry:
  %g_graph_mask_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 0
  %g_graph_mask_loaded = load ptr, ptr %g_graph_mask_offset_ptr, align 4, !vortex.uniform !32
  %g_updating_graph_mask_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 4
  %g_updating_graph_mask_loaded = load ptr, ptr %g_updating_graph_mask_offset_ptr, align 4, !vortex.uniform !32
  %g_graph_visited_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 8
  %g_graph_visited_loaded = load ptr, ptr %g_graph_visited_offset_ptr, align 4, !vortex.uniform !32
  %g_over_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 12
  %g_over_loaded = load ptr, ptr %g_over_offset_ptr, align 4, !vortex.uniform !32
  %no_of_nodes_offset_ptr = getelementptr i8, ptr %ArgBuffer, i32 16
  %no_of_nodes_loaded = load i32, ptr %no_of_nodes_offset_ptr, align 4, !vortex.uniform !32
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
  %cmp = icmp slt i32 %8, %no_of_nodes_loaded
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry1
  %arrayidx = getelementptr inbounds i8, ptr %g_updating_graph_mask_loaded, i32 %8
  %9 = load i8, ptr %arrayidx, align 1, !tbaa !33
  %tobool.not = icmp eq i8 %9, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %land.lhs.true
  %arrayidx1 = getelementptr inbounds i8, ptr %g_graph_mask_loaded, i32 %8
  store i8 1, ptr %arrayidx1, align 1, !tbaa !33
  %arrayidx2 = getelementptr inbounds i8, ptr %g_graph_visited_loaded, i32 %8
  store i8 1, ptr %arrayidx2, align 1, !tbaa !33
  store i8 1, ptr %g_over_loaded, align 1, !tbaa !33
  store i8 0, ptr %arrayidx, align 1, !tbaa !33
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry1
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull) #0

define ptr @__vx_get_kernel_callback(i32 %kernel_index) {
entry:
  switch i32 %kernel_index, label %default [
    i32 0, label %case_0
    i32 1, label %case_1
  ]

case_0:                                           ; preds = %entry
  ret ptr @BFS_1

case_1:                                           ; preds = %entry
  ret ptr @BFS_2

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
!29 = !{!"device_max_witem_sizes_0", i64 16}
!30 = !{!"device_max_witem_sizes_1", i64 16}
!31 = !{!"device_max_witem_sizes_2", i64 16}
!32 = !{!"vortex.uniform"}
!33 = !{!34, !34, i64 0}
!34 = !{!"omnipotent char", !35, i64 0}
!35 = !{!"Simple C/C++ TBAA"}
!36 = !{!37, !38, i64 4}
!37 = !{!"", !38, i64 0, !38, i64 4}
!38 = !{!"int", !34, i64 0}
!39 = !{!37, !38, i64 0}
!40 = !{!38, !38, i64 0}
