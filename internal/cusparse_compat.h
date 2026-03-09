#pragma once

#include <cusparse.h>

#if defined(CUSPARSE_VER_MAJOR) && defined(CUSPARSE_VER_MINOR) && defined(CUSPARSE_VER_PATCH)
#define CUPDLPX_CUSPARSE_GTE_13_1U1                                                                                \
    ((CUSPARSE_VER_MAJOR > 13) ||                                                                                  \
     (CUSPARSE_VER_MAJOR == 13 &&                                                                                  \
      (CUSPARSE_VER_MINOR > 1 || (CUSPARSE_VER_MINOR == 1 && CUSPARSE_VER_PATCH >= 1))))
#elif defined(CUSPARSE_VERSION)
// Fallback encoding assumption: major*1000 + minor*10 + patch
#define CUPDLPX_CUSPARSE_GTE_13_1U1 (CUSPARSE_VERSION >= 13101)
#else
#define CUPDLPX_CUSPARSE_GTE_13_1U1 0
#endif

#define CUPDLPX_HAS_SPMVOP CUPDLPX_CUSPARSE_GTE_13_1U1

#if !CUPDLPX_HAS_SPMVOP
typedef void *cusparseSpMVOpDescr_t;
typedef void *cusparseSpMVOpPlan_t;
#endif
