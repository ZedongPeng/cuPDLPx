/*
Copyright 2025 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "utils.h"

static const double HOST_ONE = 1.0;
static const double HOST_ZERO = 0.0;

bool cupdlpx_use_spmvop_by_default(void)
{
    return CUPDLPX_HAS_SPMVOP;
}

void cupdlpx_spmv_buffer_size(cusparseHandle_t sparse_handle,
                              cusparseSpMatDescr_t mat,
                              cusparseDnVecDescr_t vec_x,
                              cusparseDnVecDescr_t vec_y,
                              size_t *buffer_size)
{
#if CUPDLPX_HAS_SPMVOP
    CUSPARSE_CHECK(cusparseSpMVOp_bufferSize(sparse_handle,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             mat,
                                             vec_x,
                                             vec_y,
                                             vec_y,
                                             CUDA_R_64F,
                                             buffer_size));
#else
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           mat,
                                           vec_x,
                                           &HOST_ZERO,
                                           vec_y,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           buffer_size));
#endif
}

void cupdlpx_spmv_prepare(cusparseHandle_t sparse_handle,
                          cusparseSpMatDescr_t mat,
                          cusparseDnVecDescr_t vec_x,
                          cusparseDnVecDescr_t vec_y,
                          void *buffer,
                          cusparseSpMVOpDescr_t *descr,
                          cusparseSpMVOpPlan_t *plan)
{
#if CUPDLPX_HAS_SPMVOP
    CUSPARSE_CHECK(cusparseSpMVOp_createDescr(sparse_handle,
                                              descr,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              mat,
                                              vec_x,
                                              vec_y,
                                              vec_y,
                                              CUDA_R_64F,
                                              buffer));
    CUSPARSE_CHECK(cusparseSpMVOp_createPlan(sparse_handle, *descr, plan, NULL, 0));
#else
    (void)descr;
    (void)plan;
    CUSPARSE_CHECK(cusparseSpMV_preprocess(sparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           mat,
                                           vec_x,
                                           &HOST_ZERO,
                                           vec_y,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           buffer));
#endif
}

void cupdlpx_spmv_release(cusparseSpMVOpDescr_t descr, cusparseSpMVOpPlan_t plan)
{
#if CUPDLPX_HAS_SPMVOP
    if (descr)
    {
        CUSPARSE_CHECK(cusparseSpMVOp_destroyDescr(descr));
    }
    if (plan)
    {
        CUSPARSE_CHECK(cusparseSpMVOp_destroyPlan(plan));
    }
#else
    (void)descr;
    (void)plan;
#endif
}

void cupdlpx_spmv_execute(cusparseHandle_t sparse_handle,
                          cusparseSpMatDescr_t mat,
                          cusparseDnVecDescr_t vec_x,
                          cusparseDnVecDescr_t vec_y,
                          void *buffer,
                          cusparseSpMVOpPlan_t plan)
{
#if CUPDLPX_HAS_SPMVOP
    CUSPARSE_CHECK(cusparseSpMVOp(sparse_handle, plan, &HOST_ONE, &HOST_ZERO, vec_x, vec_y, vec_y));
#else
    (void)plan;
    CUSPARSE_CHECK(cusparseSpMV(sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                mat,
                                vec_x,
                                &HOST_ZERO,
                                vec_y,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                buffer));
#endif
}

void cupdlpx_spmv(pdhg_solver_state_t *state,
                  cusparseSpMatDescr_t mat,
                  cusparseDnVecDescr_t vec_x,
                  cusparseDnVecDescr_t vec_y,
                  void *buffer,
                  cusparseSpMVOpPlan_t plan)
{
    cupdlpx_spmv_execute(state->sparse_handle, mat, vec_x, vec_y, buffer, plan);
}
