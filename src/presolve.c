#include "presolve.h"
#include "cupdlpx.h"
#include "PSLP_sol.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>

lp_problem_t* convert_pslp_to_cupdlpx(PresolvedProblem *reduced_prob) {

    lp_problem_t *cupdlpx_prob = (lp_problem_t *)safe_malloc(sizeof(lp_problem_t));
    // TODO: handle warmstart here
    cupdlpx_prob->primal_start = NULL;
    cupdlpx_prob->dual_start = NULL;

    cupdlpx_prob->objective_constant = reduced_prob->obj_offset;
    cupdlpx_prob->objective_vector = reduced_prob->c;

    cupdlpx_prob->constraint_lower_bound = reduced_prob->lhs;
    cupdlpx_prob->constraint_upper_bound = reduced_prob->rhs;
    cupdlpx_prob->variable_lower_bound = reduced_prob->lbs;
    cupdlpx_prob->variable_upper_bound = reduced_prob->ubs;

    cupdlpx_prob->constraint_matrix_num_nonzeros = reduced_prob->nnz;
    cupdlpx_prob->constraint_matrix_row_pointers = reduced_prob->Ap;
    cupdlpx_prob->constraint_matrix_col_indices = reduced_prob->Ai;
    cupdlpx_prob->constraint_matrix_values = reduced_prob->Ax;

    cupdlpx_prob->num_variables = reduced_prob->n;
    cupdlpx_prob->num_constraints = reduced_prob->m;

    return cupdlpx_prob;
}

cupdlpx_presolve_info_t* pslp_presolve(const lp_problem_t *original_prob, const pdhg_parameters_t *params) {
    clock_t start_time = clock();
    
    cupdlpx_presolve_info_t *info = (cupdlpx_presolve_info_t*)calloc(1, sizeof(cupdlpx_presolve_info_t));
    if (!info) return NULL;

    // 1. Init Settings
    info->settings = default_settings();
    if (params->verbose) {
        info->settings->verbose = true;
    }
    // info->settings->relax_bounds = false;

    // 2. Init Presolver
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
    info->presolve_setup_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    // 3. Run Presolve
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
    // if (info->reduced_problem) lp_problem_free(info->reduced_problem);
    if (info->presolver) free_presolver(info->presolver);
    if (info->settings) free_settings(info->settings);
    free(info);
}