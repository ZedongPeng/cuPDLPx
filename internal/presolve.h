/*
 * Interface for PSLP integration with cuPDLPx
 */

#pragma once

#include "cupdlpx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Solves the LP problem using PSLP for presolving and cuPDLPx for the reduced problem.
 * * @param original_prob The original LP problem.
 * @param params Solver parameters.
 * @return cupdlpx_result_t* The solution mapped back to the original problem dimensions.
 */
cupdlpx_result_t* solve_with_pslp(
    lp_problem_t *original_prob, 
    pdhg_parameters_t *params
);

#ifdef __cplusplus
}
#endif