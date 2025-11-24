#ifndef PRESOLVE_H
#define PRESOLVE_H

#include "cupdlpx.h"
#include "PSLP_API.h" 

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Presolver *presolver;
    Settings *settings;
    lp_problem_t *reduced_problem; 
    bool problem_solved_during_presolve; 
    double presolve_time;
} cupdlpx_presolve_info_t;

cupdlpx_presolve_info_t* pslp_presolve(const lp_problem_t *original_prob, const pdhg_parameters_t *params);

cupdlpx_result_t* pslp_postsolve(cupdlpx_presolve_info_t *info, 
                                 cupdlpx_result_t *reduced_result, 
                                 const lp_problem_t *original_prob);

void cupdlpx_presolve_info_free(cupdlpx_presolve_info_t *info);

#ifdef __cplusplus
}
#endif

#endif // PRESOLVE_H