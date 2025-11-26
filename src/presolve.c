#include "presolve.h"
#include "cupdlpx.h"
#include "PSLP_inf.h"
#include "PSLP_sol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>

void sanitize_infinity_for_pslp(double* arr, int n, double pslp_inf_val) {
    for (int i = 0; i < n; ++i) {
        if (isinf(arr[i]) || fabs(arr[i]) >= pslp_inf_val) {
            arr[i] = (arr[i] > 0) ? pslp_inf_val : -pslp_inf_val;
        }
    }
}

#define CUPDLP_INF std::numeric_limits<double>::infinity()

void restore_infinity_for_cupdlpx(double* arr, int n, double pslp_inf_val) {
    for (int i = 0; i < n; ++i) {
        if (arr[i] >= pslp_inf_val * 0.99) { 
            arr[i] = INFINITY;
        }
        else if (arr[i] <= -pslp_inf_val * 0.99) {
            arr[i] = -INFINITY;
        }
    }
}

lp_problem_t* convert_pslp_to_cupdlpx(PresolvedProblem *reduced_prob) {
    matrix_desc_t desc;
    memset(&desc, 0, sizeof(matrix_desc_t)); 

    desc.m = reduced_prob->m;
    desc.n = reduced_prob->n;
    desc.fmt = matrix_csr;
    desc.zero_tolerance = 0.0; 

    desc.data.csr.nnz = reduced_prob->nnz;
    desc.data.csr.row_ptr = reduced_prob->Ap;
    desc.data.csr.col_ind = reduced_prob->Ai;
    desc.data.csr.vals = reduced_prob->Ax;

    restore_infinity_for_cupdlpx(reduced_prob->lhs, reduced_prob->m, PSLP_INF);
    restore_infinity_for_cupdlpx(reduced_prob->rhs, reduced_prob->m, PSLP_INF);
    restore_infinity_for_cupdlpx(reduced_prob->lbs, reduced_prob->n, PSLP_INF);
    restore_infinity_for_cupdlpx(reduced_prob->ubs, reduced_prob->n, PSLP_INF);

    lp_problem_t* gpu_prob = create_lp_problem(
        reduced_prob->c,
        &desc,
        reduced_prob->lhs, // constraint lower bound
        reduced_prob->rhs, // constraint upper bound
        reduced_prob->lbs, // variable lower bound
        reduced_prob->ubs, // variable upper bound
        &reduced_prob->obj_offset
    );

    return gpu_prob;
}

cupdlpx_presolve_info_t* pslp_presolve(const lp_problem_t *original_prob, const pdhg_parameters_t *params) {
    clock_t start_time = clock();
    
    cupdlpx_presolve_info_t *info = (cupdlpx_presolve_info_t*)calloc(1, sizeof(cupdlpx_presolve_info_t));
    if (!info) return NULL;

    // 1. Sanitize input data
    sanitize_infinity_for_pslp(original_prob->constraint_lower_bound, original_prob->num_constraints, PSLP_INF);
    sanitize_infinity_for_pslp(original_prob->constraint_upper_bound, original_prob->num_constraints, PSLP_INF);
    sanitize_infinity_for_pslp(original_prob->variable_lower_bound, original_prob->num_variables, PSLP_INF);
    sanitize_infinity_for_pslp(original_prob->variable_upper_bound, original_prob->num_variables, PSLP_INF);

    // 2. Init Settings
    info->settings = default_settings();
    if (params->verbose) {
        info->settings->verbose = true;
    }
    // info->settings->relax_bounds = false;

    // 3. Init Presolver
    info->presolver = new_presolver(
        original_prob->constraint_matrix_values,
        original_prob->constraint_matrix_col_indices,
        original_prob->constraint_matrix_row_pointers,
        original_prob->num_constraints,
        original_prob->num_variables,
        original_prob->constraint_matrix_num_nonzeros,
        original_prob->constraint_lower_bound,
        original_prob->constraint_upper_bound,
        original_prob->variable_lower_bound,
        original_prob->variable_upper_bound,
        original_prob->objective_vector,
        info->settings,
        true 
    );

    // 4. Run Presolve
    PresolveStatus status = run_presolver(info->presolver);
    
    if (status & INFEASIBLE || status & UNBOUNDED) {
        info->problem_solved_during_presolve = true;
        info->reduced_problem = NULL;
        if (params->verbose) printf("Problem solved by presolver (Infeasible/Unbounded).\n");
    } else {
        info->problem_solved_during_presolve = false;
        if (params->verbose) {
            printf("Presolve finished. Reduced rows: %d, cols: %d\n", 
                   info->presolver->reduced_prob->m, info->presolver->reduced_prob->n);
        }
        info->reduced_problem = convert_pslp_to_cupdlpx(info->presolver->reduced_prob);
    }

    info->presolve_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    return info;
}

cupdlpx_result_t* pslp_postsolve(cupdlpx_presolve_info_t *info, 
                                 cupdlpx_result_t *reduced_result, 
                                 const lp_problem_t *original_prob) {
    
    if (info->problem_solved_during_presolve) {
        cupdlpx_result_t *res = (cupdlpx_result_t*)calloc(1, sizeof(cupdlpx_result_t));
        
        if (info->presolver && info->presolver->stats) {
            res->presolve_stats = *(info->presolver->stats);
        }

        res->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
        res->cumulative_time_sec = info->presolve_time;
        return res;
    }

    if (!reduced_result || !info->presolver) return NULL;

    postsolve(info->presolver, 
              reduced_result->primal_solution, 
              reduced_result->dual_solution, 
              reduced_result->reduced_cost,
              reduced_result->primal_objective_value);

    cupdlpx_result_t *final_result = (cupdlpx_result_t*)malloc(sizeof(cupdlpx_result_t));
    
    *final_result = *reduced_result; 

    if (info->presolver->stats != NULL) {
        final_result->presolve_stats = *(info->presolver->stats);
    }

    final_result->primal_solution = (double*)malloc(original_prob->num_variables * sizeof(double));
    final_result->dual_solution = (double*)malloc(original_prob->num_constraints * sizeof(double));

    memcpy(final_result->primal_solution, info->presolver->sol->x, original_prob->num_variables * sizeof(double));
    memcpy(final_result->dual_solution, info->presolver->sol->y, original_prob->num_constraints * sizeof(double));
    final_result->primal_objective_value = info->presolver->sol->obj;
    final_result->cumulative_time_sec += info->presolve_time;

    return final_result;
}

void cupdlpx_presolve_info_free(cupdlpx_presolve_info_t *info) {
    if (!info) return;
    if (info->reduced_problem) lp_problem_free(info->reduced_problem);
    if (info->presolver) free_presolver(info->presolver);
    if (info->settings) free_settings(info->settings);
    free(info);
}