///This file computes the sol of Ax = b with a preconditioned CG (preconditioned with a switching precision Cholesky)
/// @author Sudhanva Kulkarni

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include "flops.h"
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/float8_iee_p.hpp>
#include <tlapack/plugins/eigen_bfloat.hpp>
#include <tlapack/plugins/eigen_half.hpp>
#include "opcounts.hpp"
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
#include "pivoted_cholesky.hpp"
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
#include <fstream>
#include "python_utils.hpp"


std::ofstream myfile("e5m2_error_f_cond.csv");
std::ofstream timefile("time.txt");

template<typename F0, typename F1, typename T2, typename T3, typename T4, typename T5, typename T6>
CG_IR(size_t n, double scale, float cond, factorization_type fact_type, double switching_val, double tolerance, double conv_thresh, bool is_symmetric, int p,  int variant = 0, int num_iter_1 = 0, int num_total_iter = 5, kernel_type precond_mode = kernel_type::LEFT_LU, int max_CG_iter = 20, int inner_num_gmres_iter = 20, int num_IR_iter=20, refinement_type method = refinement_type::CG_IR, int block_size=128, int stopping_pos=99999)  {
    using matrix_t = tlapack::LegacyMatrix<ml_dtypes::float8_ieee_p<4>>;
    using block_mat_t = tlapack::BlockMatrix<ml_dtypes::float8_ieee_p<4>>;
    using real_t = tlapack::real_type<ml_dtypes::float8_ieee_p<4>>;
    using idx_t = size_t;
    using range = pair<idx_t, idx_t>;

    double d_conv_bound = sqrt(static_cast<double>(n)) * std::pow(2, -53);
    double s_conv_bound = sqrt(static_cast<double>(n)) * std::pow(2, -24);

    double total_time = 0.0;   

    vector<int> left_piv(n);
    vector<int> right_piv(n);
    // Create the n-by-n matrix A in precision float
    std::vector<float> A_(n * n);
    std::vector<int> A_exp(n, 0);
    //tlapack::BlockMatrix<float> A(n, n, A_.data(), n, A_exp.data(), n);
    tlapack::LegacyMatrix<float, idx_t> A(n, n, A_.data(), n);

    // FG matrix is "for generation" so it will be stored in double precision
    std::vector<double> FG_(n * n);
    tlapack::LegacyMatrix<double, idx_t> FG(n, n, FG_.data(), n);

    // FG matrix is "for generation" so it will be stored in double precision
    std::vector<double> FG_d_(n * n);
    tlapack::LegacyMatrix<double, idx_t> FG_d(n, n, FG_d_.data(), n);

    //we'll also need a single precision copy of A
    std::vector<T6> A_F_(n * n);
    tlapack::LegacyMatrix<T6, idx_t> A_F(n, n, A_F_.data(), n);

    //now we'll store A in precision T2 (this is the intermediate precision we'll apply the first IR in)
    std::vector<T2> A_B_(n*n);
    tlapack::LegacyMatrix<T2, idx_t> A_B(n,n, A_B_.data(), n);

    //next we need a precision T3 which is the precision on which we'll get the initial solution
    std::vector<T3> A_C_(n*n);
    tlapack::LegacyMatrix<T3, idx_t> A_C(n, n, A_C_.data(), n);

    //Zero matrix for whenever we need to initailize a matrix to 0
    std::vector<float> Zero_(n * n);
    tlapack::LegacyMatrix<float, idx_t> Zero(n, n, Zero_.data(), n);


    float true_cond;
    //construct the matrix in desired precision
    constructMatrix<float>(n, cond, std::ceil(cond/static_cast<float>(5)) > n-1 ? n-1 : std::ceil(cond/static_cast<float>(5)) , false, FG, p, is_symmetric, true_cond);
   
    



    std::vector<float> S_scal(n, 0.0);
    for (size_t i = 0; i < n; i++){
        S_scal[i] = 1.0;
    }
    std::vector<float> R_scal(n, 0.0);
    for (size_t i = 0; i < n; i++){
        R_scal[i] = 1.0;
    }

        

    //     //next we need to scale by a parameter theta
    double maxA = tlapack::lange(tlapack::Norm::Max, FG);

        
    double normA = tlapack::lange(tlapack::Norm::Inf, FG);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            FG(i,j) = scale*FG(i,j);
        }
    }   

    

    tlapack::lacpy(tlapack::Uplo::General, FG, A);
    for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < n; ++i){
            A_F(i,j) = static_cast<T6>(FG(i,j));
            A_B(i,j) = static_cast<T2>(FG(i,j));
            A_C(i,j) = static_cast<T3>(FG(i,j));
            Zero(i,j) = 0.0;
        }
    }

     //first generate the solution vector. Since we want to reach a double precision solution, we'll generate the solution vector in double precision
    std::vector<double> x_(n);
    tlapack::LegacyVector<double, idx_t> x(n, x_.data());

    //we'll need a copy of x in each precision for calls to gemm, etc
    std::vector<double> x_1_(n);
    tlapack::LegacyVector<double, idx_t> x_1(n, x_1_.data());
    std::vector<T2> x_2_(n);
    tlapack::LegacyVector<T2, idx_t> x_2(n, x_2_.data());
    std::vector<T3> x_3_(n);
    tlapack::LegacyVector<T3, idx_t> x_3(n, x_3_.data());

    //a copy of x in float just because
    std::vector<double> x_f_(n);
    tlapack::LegacyVector<double, idx_t> x_f(n, x_f_.data());

    //b will be genrated in double and we'll store low precision copies for it just like x
    std::vector<double> b_(n);
    tlapack::LegacyVector<double, idx_t> b(n, b_.data());

    std::vector<double> b_1_(n);
    tlapack::LegacyVector<double, idx_t> b_1(n, b_1_.data());

    std::vector<T2> b_2_(n);
    tlapack::LegacyVector<T2, idx_t> b_2(n, b_2_.data());

    // Generate b in T3
    std::vector<T3> b_3_(n);
    tlapack::LegacyVector<T3, idx_t> b_3(n, b_3_.data());

    // Generate b in float
    std::vector<double> b_f_(n);
    tlapack::LegacyVector<double, idx_t> b_f(n, b_f_.data());


    //bd for "true sol" -- may need to change this to quad
    std::vector<double> bd_(n);
    tlapack::LegacyVector<double, idx_t> bd(n, bd_.data());

    std::vector<double> r_(n);
    tlapack::LegacyVector<double, idx_t> r(n, r_.data());

    std::vector<double> r_1_(n);
    tlapack::LegacyVector<double, idx_t> r_1(n, r_1_.data());

    std::vector<T2> r_2_(n);
    tlapack::LegacyVector<T2, idx_t> r_2(n, r_2_.data());

    std::vector<T3> r_3_(n);
    tlapack::LegacyVector<T3, idx_t> r_3(n, r_3_.data());

    std::vector<T6> r_f_(n);
    tlapack::LegacyVector<T6, idx_t> r_f(n, r_f_.data());

    std::vector<double> solved_r_(n);
    tlapack::LegacyVector<double, idx_t> solved_r(n, solved_r_.data());

    std::vector<double> solved_r_1_(n);
    tlapack::LegacyVector<double, idx_t> solved_r_1(n, solved_r_1_.data());

    std::vector<T2> solved_r_2_(n);
    tlapack::LegacyVector<T2, idx_t> solved_r_2(n, solved_r_2_.data());

    std::vector<T3> solved_r_3_(n);
    tlapack::LegacyVector<T3, idx_t> solved_r_3(n, solved_r_3_.data());

    std::vector<T6> solved_r_f_(n);
    tlapack::LegacyVector<T6, idx_t> solved_r_f(n, solved_r_f_.data());



    std::vector<double> be_1_(n);
    tlapack::LegacyVector<double, idx_t> be_1(n, be_1_.data());

    std::vector<double> be_1_1_(n);
    tlapack::LegacyVector<double, idx_t> be_1_1(n, be_1_1_.data());

    std::vector<T2> be_1_2_(n);
    tlapack::LegacyVector<T2, idx_t> be_1_2(n, be_1_2_.data());

    std::vector<T3> be_1_3_(n);
    tlapack::LegacyVector<T3, idx_t> be_1_3(n, be_1_3_.data());

    
    std::vector<T6> be_1_f_(n);
    tlapack::LegacyVector<T6, idx_t> be_1_f(n, be_1_f_.data());


    for( int i = 0; i < n; i++) {
        b[i] = static_cast<double>((-0.5*(static_cast<double>(rand()))/static_cast<double>(RAND_MAX)));
        b[i] += static_cast<double>((static_cast<double>(rand())/static_cast<double>(RAND_MAX)));
        b_f[i] = static_cast<double>(b[i]);
        b_1[i] = static_cast<double>(b[i]);
        b_2[i] = static_cast<T2>(b[i]);
        b_3[i] = static_cast<T3>(b[i]);
        be_1[i] = (i == 0 ? 1.0 : 0.0);
        be_1_f[i] = static_cast<T6>(be_1[i]);
        be_1_1[i] = static_cast<double>(be_1[i]);
        be_1_2[i] = static_cast<T2>(be_1[i]);
        be_1_3[i] = static_cast<T3>(be_1[i]);
        bd[i] = static_cast<double>(b[i]);
    }

    //perform LU on A and FG
    std::vector<float> LL_(n * n);
    std::vector<int> LL_exp(n, 0);
    //tlapack::BlockMatrix<float> LU(n, n, LU_.data(), n, LU_exp.data(), n);
    tlapack::LegacyMatrix<float, idx_t> LL(n, n, LL_.data(), n);

    //keep copies in float and the second precision
    std::vector<T6> LL_copy_(n * n);
    tlapack::LegacyMatrix<T6, idx_t> LL_copy(n, n, LL_copy_.data(), n);

    std::vector<T2> LL_copy_B_(n * n);
    tlapack::LegacyMatrix<T2, idx_t> LL_copy_B(n, n, LL_copy_B_.data(), n);

    std::vector<T3> LL_copy_C_(n * n);
    tlapack::LegacyMatrix<T3, idx_t> LL_copy_C(n, n, LL_copy_C_.data(), n);

    std::vector<T4> LL_copy_D_(n * n);
    tlapack::LegacyMatrix<T4, idx_t> LL_copy_D(n, n, LL_copy_D_.data(), n);

    std::vector<T5> LL_copy_E_(n * n);
    tlapack::LegacyMatrix<T5, idx_t> LL_copy_E(n, n, LL_copy_E_.data(), n);


    std::vector<double> LL_double_(n * n);
    tlapack::LegacyMatrix<double, idx_t> LL_double(n, n, LL_double_.data(), n);

    //first we need to get the soln in double-
    tlapack::lacpy(tlapack::GENERAL, FG, LL_double);
    tlapack::lacpy(tlapack::GENERAL, A, LL);

    int info = pivoted_cholesky(LL, left_piv, right_piv, block_size, tolerance);
    if(info != 0) {
        cout << " Cholesky in low precision failed" << endl;
        return -1.0;
    }

    int infotoo = cholesky_kernel(LL_double);
    if(infotoo != 0) {
        cout << " Matrix is not SPD" << endl;
        return -1.0;
    }



    //compute true soln in double
    tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::NO_TRANS, Diag::NonUnit, LL_double, bd);
    tlapack::trsv(tlapack::UPPER_TRIANGLE, tlapack::TRANSPOSE, Diag::NonUnit, LL_double, bd);

    //"true sol" in bd, now compute init sol with custom Cholesky

    for (int i = 0; i < n; i++)
    {
        if (left_piv[i] != i)
        {
            auto tmp = b_3[left_piv[i]];
            b_3[left_piv[i]] = b_3[i];
            b_3[i] = tmp;
        }
    }
    tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::NO_TRANS, Diag::NonUnit, LL, b_3);
    tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::TRANSPOSE, Diag::NonUnit, LL, b_3);

    for (int i = 0; i < n; i++)
    {
        if (right_piv[i] != i)
        {
            auto tmp = b_3[right_piv[i]];
            b_3[right_piv[i]] = b_3[i];
            b_3[i] = tmp;
        }
    }


    for(int i = 0; i < n; i++) x[i] = b_3[i];
    for(int i = 0; i < n; i++) r_3[i] = b[i];

    
    // We now need to use r_3 as thr RHS for CG

   
    // Allocate necessary vectors
    std::vector<double> r(n);  // Residual vector
    std::vector<double> z(n);  // Preconditioned vector
    std::vector<double> p(n);  // Search direction
    std::vector<double> temp_p(n);
    std::vector<double> Ap(n); // A * p

    auto nrmb = tlapack::nrm2(b);

    auto norm_x = 0.0;

    
    
    

    int count = 0;

    for(int j = 0; j < max_CG_iter; j++) {
        // r = b - A * x (initial residual, but with x = 0, r = b)
        for(int i = 0; i < n; i++)  r[i] = x[i];
        tlapack::gemv(tlapack::NO_TRANS, -1.0, FG, b, r);       // r = b - Ax

        //precondition r
        for (int i = 0; i < n; i++) {
            if( left_piv[i] != i) {
                auto tmp = r[right_piv[i]];
                r[right_piv[i]] = r[i];
                r[i] = tmp;
            }
        }
        tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::NO_TRANS, Diag::NonUnit, LL, r);


        double old_rho = tlapack::nrm2(r);
        double new_rho = old_rho;
    
    for (int iter = 0; iter < n; ++iter) {

        if(new_rho <= conv_thresh*nrmb) {
            break; 
        }

        if(iter == 0) {
            for(int i = 0; i < n; i++) p[i] = r[i];
        } else {
            tlapack::scal(new_rho/old_rho, p);
            tlapack::axpy(1.0,r, p);
        }
        
        for(int i = 0; i < n; i++) temp_p[i] = p[i];

        // Ap = A * p
        for (int i = 0; i < n; i++)
        {
            if (right_piv[i] != i)
            {
                auto tmp = temp_p[right_piv[i]];
                temp_p[right_piv[i]] = temp_p[i];
                temp_p[i] = tmp;
            }
        }
        tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::TRANSPOSE, Diag::NonUnit, LL, temp_p);
        tlapack::gemv(tlapack::NO_TRANS, 1.0, A, temp_p, 0.0, Ap);
        tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::NO_TRANS, Diag::NonUnit, LL, Ap);

        for (int i = 0; i < n; i++)
        {
            if (left_piv[i] != i)
            {
                auto tmp = Ap[left_piv[i]];
                Ap[left_piv[i]] = Ap[i];
                Ap[i] = tmp;
            }
        }




        // alpha = rsold / (p^T * inv(L)* A* inv(L) * p)
        double pAp = tlapack::dot(Ap, p);
        double alpha = new_rho / pAp;

        // x = x + alpha * p
        tlapack::axpy(alpha, p, x);

        // r = r - alpha * A * p
        tlapack::axpy(-alpha, Ap, r);

        norm_x =  inf_norm(x);

        myfile << count << "," << inf_norm(r)/(normA*norm_x) << std::endl;

        // Check convergence: ||r|| < tol
        double rnorm = tlapack::nrm2(r);
        if (rnorm < d_conv_bound) {
            std::cout << "Converged to double in " << iter + 1 << " iterations." << std::endl;
            break;
        } else if(rnorm < s_conv_bound) {
            std::cout << "Converged to Single in " << iter + 1 << " iterations." << endl;
        }


        old_rho = new_rho;
        new_rho = tlapack::nrm2(r);

        count++;
    }

    //now need to adjust the soln by trmv and permute
    for(int i = 0; i < n; i++) solved_r[i] = r[i];
    for (int i = 0; i < n; i++)
        {
            if (right_piv[i] != i)
            {
                auto tmp = solved_r[right_piv[i]];
                solved_r[right_piv[i]] = solved_r[i];
                solved_r[i] = tmp;
            }
        }

    tlapack::trmv(Uplo::Lower, Op::Trans, Diag::NonUnit, LL, solved_r);

    tlapack::axpy(1.0, solved_r, x);

    }



    double norm_x = inf_norm(x);
    tlapack::gemv(tlapack::NO_TRANS, 1.0, FG, x, -1.0, b);
    double res_norm = inf_norm(b)/(normA*norm_x);
    std::cout << "backward err ; " << res_norm << std::endl;

    for(int i = 0; i < n; i++) x[i] = x[i] - bd[i];
    std::cout << "forward error is : " << inf_norm(x)/inf_norm(bd) << std::endl;


    timefile << total_time << std::endl;

    if(res_norm == 0.0) res_norm = 999999.0;
    std::cout << "backward err ; " << res_norm << std::endl;
    std::cout << "total number of higher prec gmres loops : " << num_higher_prec_gmres_loops << std::endl;
    std::cout << "total number of lower prec gmres loops : " << num_lower_prec_gmres_loops << std::endl;
    return res_norm;






}

int main(int argc, char** argv) {

     //matrix params
    int n;
    float cond;
    bool is_symmetric;

    //fact params
    int block_size, stopping_pos;
    factorization_type fact_type;
    string lowest_prec, highest_prec;
    bool is_rank_revealing;
    pivoting_scheme pivot_scheme;
    int num_precisions;
    bool use_microscal;
    double switching_val, scaling_factor, tolerance;



    //refinement params
    int max_gmres_iter, num_iter_1, total_num_iter, num_IR_iter;
    int inner_gmres_num = 0;
    refinement_type refinement_method;
    kernel_type precond_kernel;
    ortho_type arnoldi_subroutine;


    double er3 = 0.0;
    std::cout << std::scientific << std::endl;

    std::ifstream settings_file("settings.json", std::ifstream::binary);
    nlohmann::json settings = nlohmann::json::parse(settings_file);
   
   
    //set  arguments from settings.json
    std::cout << "setting program vars from JSON" << std::endl;
    set_matrix_params(n, cond, is_symmetric, settings);
    std::cout << "set matrix params" << std::endl;
    set_factorization_params(fact_type, lowest_prec, highest_prec, is_rank_revealing, pivot_scheme, num_precisions, block_size, use_microscal, stopping_pos, switching_val, scaling_factor, tolerance, settings);
    std::cout << "set factorization params" << std::endl;
    set_refinement_settings(max_gmres_iter, num_iter_1, total_num_iter, refinement_method, precond_kernel, num_IR_iter, inner_gmres_num, arnoldi_subroutine, settings);
    std::cout << "set refinement params" << std::endl;

    std::cout << "beginning factorization" << std::endl;
    er3 += CG_IR<ml_dtypes::float8_ieee_p<6>,Eigen::half, double, double, double, double, double>(n, scaling_factor, static_cast<float>(cond), fact_type, switching_val, tolerance, is_symmetric, 999, 0, num_iter_1, total_num_iter, precond_kernel, max_gmres_iter, inner_gmres_num, num_IR_iter, refinement_method, block_size, stopping_pos);    
    
    
    bool verified = abs(er3) < 1e-8;
    FILE *fp = fopen("./log.txt", "w");
    fputs(verified ? "true\n" : "false\n", fp);
    fprintf(fp, "%20.13E\n", static_cast<double>(er3));
    std::cout << "err3 : " << er3 << std::endl;
    return 0;

}