/// This file computes the sol of Ax = b with a preconditioned CG (preconditioned with a switching precision Cholesky)
///  @author Sudhanva Kulkarni

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp> 

#include <tlapack/plugins/eigen_bfloat.hpp>
#include <tlapack/plugins/eigen_half.hpp>
#include <tlapack/plugins/lo_float_sci.hpp>
#include <tlapack/blas/axpy.hpp>
#include <tlapack/blas/dot.hpp>
#include <tlapack/blas/nrm2.hpp>
#include <tlapack/blas/scal.hpp>
#include <tlapack/blas/trsv.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/gemv.hpp>
#include <tlapack/blas/ger.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/blas/trmv.hpp>

#include "gemms.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <string>
#include "GMRES.hpp"
#include "scalings.hpp"
#include "json.hpp"
#include "getrf_random.hpp"
#include "python_utils.hpp"
#include "json_utils.hpp"
#include "num_ops.hpp"
#include "pivoted_cholesky.hpp"
#include "CG_IR.hpp"
//#include "sparse_structs.hpp"






int main(int argc, char **argv)
{

    // matrix params
    int n;
    float cond, work_factor;
    bool is_symmetric, diag_dom;

    // fact params
    int block_size, stopping_pos;
    factorization_type fact_type;
    chol_mod chol_modif;
    string lowest_prec, highest_prec;
    bool is_rank_revealing;
    pivoting_scheme pivot_scheme;
    int num_precisions;
    bool use_microscal;
    double switching_val, scaling_factor, tolerance, conv_thresh;
    float dropping_prob;
    // refinement params
    int max_gmres_iter, num_iter_1, total_num_iter, num_IR_iter;
    int inner_gmres_num = 0;
    refinement_type refinement_method;
    kernel_type precond_kernel;
    ortho_type arnoldi_subroutine;

    double er3 = 0.0;
    std::cout << std::scientific << std::endl;

    std::ifstream settings_file("settings.json", std::ifstream::binary);
    nlohmann::json settings = nlohmann::json::parse(settings_file);

    // set  arguments from settings.json
    std::cout << "setting program vars from JSON" << std::endl;
    set_matrix_params(n, cond, is_symmetric, diag_dom, work_factor, settings);
    std::cout << "set matrix params" << std::endl;
    set_factorization_params(fact_type, chol_modif, lowest_prec, highest_prec, is_rank_revealing, pivot_scheme, num_precisions, block_size, use_microscal, stopping_pos, switching_val, scaling_factor, tolerance, dropping_prob, settings);
    std::cout << "set factorization params" << std::endl;
    set_refinement_settings(max_gmres_iter, num_iter_1, total_num_iter, refinement_method, precond_kernel, num_IR_iter, inner_gmres_num, arnoldi_subroutine, conv_thresh, settings);
    std::cout << "set refinement params" << std::endl;

    std::cout << "beginning factorization" << std::endl;
    float true_cond;
    // construct the matrix in desired precision
    std::vector<float> FG_(n * n);
    tlapack::LegacyMatrix<float, size_t> FG(n, n, FG_.data(), n);
    constructMatrix<float>(n, cond, std::ceil(cond / static_cast<float>(5)) > n - 1 ? n - 1 : std::ceil(cond / static_cast<float>(5)), false, FG, prec, is_symmetric, diag_dom, true_cond);

    cout << "true condition number is : " << true_cond << "\n";

    bool report_imporovement = false;
    er3 += CG_IR<float, Eigen::half, double, double, double, double, double>(FG, n ,report_improvement, scaling_factor, static_cast<float>(cond), fact_type, switching_val, tolerance, conv_thresh, is_symmetric, diag_dom, dropping_prob, chol_modif, 999, 0, max_gmres_iter, inner_gmres_num, num_IR_iter, refinement_method, block_size, stopping_pos);

    bool verified = abs(er3) < 1e-8;
    FILE *fp = fopen("./log.txt", "w");
    fputs(verified ? "true\n" : "false\n", fp);
    fprintf(fp, "%20.13E\n", static_cast<double>(er3));
    std::cout << "err3 : " << er3 << std::endl;
    return 0;
}
