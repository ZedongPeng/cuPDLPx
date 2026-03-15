#pragma once

#include <cusparse.h>

// cusparseSpMVOp_bufferSize was introduced in cuSPARSE 12.7.3 (CUDA 13.1 Update 1).
// CUSPARSE_VERSION encoding: major*1000 + minor*100 + patch.
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12703
#define CUPDLPX_HAS_SPMVOP 1
#else
#define CUPDLPX_HAS_SPMVOP 0
#endif

#if !CUPDLPX_HAS_SPMVOP
typedef void *cusparseSpMVOpDescr_t;
typedef void *cusparseSpMVOpPlan_t;
#endif
