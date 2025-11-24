#include "cupdlpx.h"
#include "PSLP_API.h"
#include "PSLP_inf.h"
#include "PSLP_sol.h"
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>

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

    // Map CSR data (Both PSLP and cuPDLPx use 0-based int and double, pointers can be passed directly)
    desc.data.csr.nnz = reduced_prob->nnz;
    desc.data.csr.row_ptr = reduced_prob->Ap;
    desc.data.csr.col_ind = reduced_prob->Ai;
    desc.data.csr.vals = reduced_prob->Ax;

    printf("DEBUG: Converting to cuPDLPx. Size %dx%d, NNZ %d\n", desc.m, desc.n, desc.data.csr.nnz);

    double obj_const = 0.0; 

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
        &obj_const
    );

    return gpu_prob;
}

/*
 * Core function: Solver entry point with presolve
 */
cupdlpx_result_t* solve_with_pslp(
    lp_problem_t *original_prob, 
    pdhg_parameters_t *params
) {
    sanitize_infinity_for_pslp(original_prob->constraint_lower_bound, original_prob->num_constraints, PSLP_INF);
    sanitize_infinity_for_pslp(original_prob->constraint_upper_bound, original_prob->num_constraints, PSLP_INF);
    sanitize_infinity_for_pslp(original_prob->variable_lower_bound, original_prob->num_variables, PSLP_INF);
    sanitize_infinity_for_pslp(original_prob->variable_upper_bound, original_prob->num_variables, PSLP_INF);
    // ---------------------------------------------------------
    // 1. Initialize PSLP and run
    // ---------------------------------------------------------
    Settings *settings = default_settings();
    if (params->verbose) {
        settings->verbose = true;
    }
    Presolver *presolver = new_presolver(
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
        settings,
        true // Input is in CSR format
    );

    if (!presolver) {
        fprintf(stderr, "PSLP Initialization Failed.\n");
        free_settings(settings);
        return NULL;
    }

    PresolveStatus status = run_presolver(presolver);
    
    // Handle cases where presolve directly detects infeasible/unbounded status
    if (status & INFEASIBLE || status & UNBOUNDED) {
        printf("Problem solved by presolver (Infeasible/Unbounded).\n");
        free_presolver(presolver);
        free_settings(settings);
        return NULL; 
    }

    // ---------------------------------------------------------
    // 2. Pass the reduced problem to cuPDLPx (GPU)
    // ---------------------------------------------------------
    printf("Presolve finished. Reduced size: %d rows, %d cols\n", 
           presolver->reduced_prob->m, presolver->reduced_prob->n);

    lp_problem_t *reduced_gpu_prob = convert_pslp_to_cupdlpx(presolver->reduced_prob);

    // Call cuPDLPx to solve
    cupdlpx_result_t *reduced_result = solve_lp_problem(reduced_gpu_prob, params);

    // ---------------------------------------------------------
    // 3. Postsolve (Recover original solution)
    // ---------------------------------------------------------
    int n_red = presolver->reduced_prob->n;
    double *z_dummy = (double*)calloc(n_red, sizeof(double)); 
    // TODO: Strictly speaking, z = c - A'y should be calculated, but if it doesn't affect primal recovery, 0 can be used
    // for (int j = 0; j < n_red; ++j) {
    //     z_dummy[j] = presolver->reduced_prob->c[j]; // z = c
    // }

    postsolve(presolver, 
              reduced_result->primal_solution, 
              reduced_result->dual_solution, 
              z_dummy, 
              reduced_result->primal_objective_value);

    // ---------------------------------------------------------
    // 4. Construct final result
    // ---------------------------------------------------------
    // PSLP results are stored in presolver->sol, need to copy them back to cupdlpx_result_t
    cupdlpx_result_t *final_result = (cupdlpx_result_t*)malloc(sizeof(cupdlpx_result_t));
    
    // Shallow copy statistics (iter count, time, etc.)
    *final_result = *reduced_result; 
    
    // Reallocate memory for original solution vectors
    final_result->primal_solution = (double*)malloc(original_prob->num_variables * sizeof(double));
    final_result->dual_solution = (double*)malloc(original_prob->num_constraints * sizeof(double));

    // Copy data from presolver->sol
    memcpy(final_result->primal_solution, presolver->sol->x, original_prob->num_variables * sizeof(double));
    memcpy(final_result->dual_solution, presolver->sol->y, original_prob->num_constraints * sizeof(double));
    
    // Update final objective function value
    final_result->primal_objective_value = presolver->sol->obj;
    // TODO: Dual objective value recovery

    // ---------------------------------------------------------
    // 5. Cleanup
    // ---------------------------------------------------------
    free(z_dummy);
    lp_problem_free(reduced_gpu_prob);
    cupdlpx_result_free(reduced_result); // Free intermediate results
    free_presolver(presolver);
    free_settings(settings);

    return final_result;
}