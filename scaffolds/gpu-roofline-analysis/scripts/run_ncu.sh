#!/usr/bin/env bash
# Profile all kernels with Nsight Compute and export CSV for the notebook
set -euo pipefail

BUILD=./build
OUT=./data
mkdir -p "$OUT"

KERNELS=(sgemm_naive sgemm_tiled sgemm_tensor_core)

for k in "${KERNELS[@]}"; do
  echo "Profiling $k ..."
  ncu \
    --metrics \
      l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__cycles_elapsed.avg,\
sm__cycles_elapsed.avg.per_second,\
smsp__warps_active.avg.pct_of_peak_sustained_active \
    --csv \
    --log-file "$OUT/${k}.csv" \
    "$BUILD/$k" 2>/dev/null || true
  echo "  → $OUT/${k}.csv"
done

echo "Done. Run: jupyter notebook notebooks/roofline_analysis.ipynb"
