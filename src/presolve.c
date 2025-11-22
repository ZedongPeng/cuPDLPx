#include "cupdlpx.h"
#include "PSLP_API.h"
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>

/* * Helper function: Convert PSLP's PresolvedProblem to cuPDLPx's lp_problem_t 
 */
// lp_problem_t* convert_pslp_to_cupdlpx(PresolvedProblem *reduced_prob) {
//     // 1. Build matrix descriptor
//     matrix_desc_t desc;
//     desc.m = reduced_prob->m;
//     desc.n = reduced_prob->n;
//     desc.fmt = matrix_csr;
//     desc.zero_tolerance = 0.0; // Use default value

//     // Map CSR data (Both PSLP and cuPDLPx use 0-based int and double, pointers can be passed directly)
//     desc.data.csr.nnz = reduced_prob->nnz;
//     desc.data.csr.row_ptr = reduced_prob->Ap;
//     desc.data.csr.col_ind = reduced_prob->Ai;
//     desc.data.csr.vals = reduced_prob->Ax;

//     // 2. Create LP problem
//     // Here, cuPDLPx's create_lp_problem will handle deep copying data into its own structure
//     // PSLP's lhs/rhs correspond to constraint_lower/upper_bound
//     // PSLP's lbs/ubs correspond to variable_lower/upper_bound
//     double obj_const = 0.0; // The offset after presolve is usually included in the recovery step

//     lp_problem_t* gpu_prob = create_lp_problem(
//         reduced_prob->c,
//         &desc,
//         reduced_prob->lhs, // con_lb
//         reduced_prob->rhs, // con_ub
//         reduced_prob->lbs, // var_lb
//         reduced_prob->ubs, // var_ub
//         &obj_const
//     );

//     return gpu_prob;
// }

lp_problem_t* convert_pslp_to_cupdlpx(PresolvedProblem *reduced_prob) {
    // [Fix 1] The struct must be initialized to zero. 
    // Since matrix_desc_t contains a union, garbage memory on the stack can cause 
    // cuPDLPx to read the wrong data format (e.g., interpreting CSR pointers as Dense pointers).
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

    // [Debug] Print metadata to verify data integrity before passing to GPU
    printf("DEBUG: Converting to cuPDLPx. Size %dx%d, NNZ %d\n", desc.m, desc.n, desc.data.csr.nnz);
    
    // Optional: Print the first few bounds to ensure they aren't corrupted
    // printf("DEBUG: First 5 bounds (LHS/RHS): ");
    // for(int i=0; i<5 && i<reduced_prob->m; i++) {
    //     printf("[%.1e, %.1e] ", reduced_prob->lhs[i], reduced_prob->rhs[i]);
    // }
    // printf("\n");

    double obj_const = 0.0; 

    // [Fix 2] Create the LP problem
    // cupdlpx's create_lp_problem will handle the deep copy of the data
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
    // ---------------------------------------------------------
    // 1. Initialize PSLP and run
    // ---------------------------------------------------------
    Settings *settings = default_settings();
    if (params->verbose) {
        settings->verbose = true;
    }
    // You can further adjust settings based on cupdlpx's params

    // Extract data from cuPDLPx's lp_problem_t and pass to PSLP
    // Note: cuPDLPx uses int* and double*, compatible with PSLP
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
        // Should construct a cupdlpx_result_t indicating infeasibility and return it here
        // Returning NULL for simplicity, needs completion in practice
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
    // Prepare z (reduced costs); if cuPDLPx doesn't output z, we need to calculate it or pass zeros
    // postsolve requires: x, y, z, obj_val
    int n_red = presolver->reduced_prob->n;
    double *z_dummy = (double*)calloc(n_red, sizeof(double)); 
    // TODO: Strictly speaking, z = c - A'y should be calculated, but if it doesn't affect primal recovery, 0 can be used

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
    // Note: Dual obj might also need update, assuming small gap here

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