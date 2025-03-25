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
//#include "sparse_structs.hpp"




std::ofstream myfile("e5m2_error_f_cond.csv");
std::ofstream other_file("e5m2_error_e_cond.csv");
std::ofstream timefile("time.txt");

template <typename F0, typename F1, typename T2, typename T3, typename T4, typename T5, typename T6>
double CG_IR(int n, double scale, float cond, factorization_type fact_type, double switching_val, double tolerance, double conv_thresh, float work_factor, bool is_symmetric, bool diag_dom, float dropping_prob, chol_mod chol_modif, int prec, int variant = 0, int num_iter_1 = 0, int num_total_iter = 5, kernel_type precond_mode = kernel_type::LEFT_LU, int max_CG_iter = 20, int inner_num_gmres_iter = 20, int num_IR_iter = 20, refinement_type method = refinement_type::CG_IR, int block_size = 128, int stopping_pos = 99999)
{

    using idx_t = size_t;
    using range = pair<idx_t, idx_t>;



    // auto TR = lo_float::float6_p<3>(1.0);

    double d_conv_bound = sqrt(static_cast<double>(n)) * std::pow(2, -32);
    double s_conv_bound = sqrt(static_cast<double>(n)) * std::pow(2, -24);

    n_flops mixed_prec_flops(work_factor); mixed_prec_flops.reset();
    n_flops reg_flops(work_factor); reg_flops.reset();
    n_flops mixed_fact_flops(work_factor);  mixed_fact_flops.reset();
    n_flops cholesky_flops(work_factor); cholesky_flops.reset();

    //cholesky flops is just n^3/3
    auto nn = (long) n;
    cholesky_flops.add_double_flops(nn*nn*(nn/3));



    

    double total_time = 0.0;

    vector<int> left_piv(n);
    vector<int> right_piv(n);
    for (int i = 0; i < n; i++)
    {
        left_piv[i] = i;
        right_piv[i] = i;
    }
    // Create the n-by-n matrix A in precision float
    std::vector<float> A_(n * n);
    std::vector<int> A_exp(n, 0);
    // tlapack::BlockMatrix<float> A(n, n, A_.data(), n, A_exp.data(), n);
    tlapack::LegacyMatrix<float, idx_t> A(n, n, A_.data(), n);

    // FG matrix is "for generation" so it will be stored in double precision
    std::vector<double> FG_(n * n);
    tlapack::LegacyMatrix<double, idx_t> FG(n, n, FG_.data(), n);

    // FG matrix is "for generation" so it will be stored in double precision
    std::vector<double> FG_d_(n * n);
    tlapack::LegacyMatrix<double, idx_t> FG_d(n, n, FG_d_.data(), n);

    // we'll also need a single precision copy of A
    std::vector<T6> A_F_(n * n);
    tlapack::LegacyMatrix<T6, idx_t> A_F(n, n, A_F_.data(), n);

    // now we'll store A in precision T2 (this is the intermediate precision we'll apply the first IR in)
    std::vector<T2> A_B_(n * n);
    tlapack::LegacyMatrix<T2, idx_t> A_B(n, n, A_B_.data(), n);

    // next we need a precision T3 which is the precision on which we'll get the initial solution
    std::vector<T3> A_C_(n * n);
    tlapack::LegacyMatrix<T3, idx_t> A_C(n, n, A_C_.data(), n);

    // Zero matrix for whenever we need to initailize a matrix to 0
    std::vector<float> Zero_(n * n);
    tlapack::LegacyMatrix<float, idx_t> Zero(n, n, Zero_.data(), n);

    std::vector<float> X_(n * n);
    tlapack::LegacyMatrix<float, idx_t> X(n, n, X_.data(), n);

    float true_cond;
    // construct the matrix in desired precision
    constructMatrix<float>(n, cond, std::ceil(cond / static_cast<float>(5)) > n - 1 ? n - 1 : std::ceil(cond / static_cast<float>(5)), false, FG, prec, is_symmetric, diag_dom, true_cond);

    cout << "true condition number is : " << true_cond << "\n";

    std::vector<float> S_scal(n, 0.0);
    for (size_t i = 0; i < n; i++)
    {
        S_scal[i] = 1.0;
    }
    std::vector<float> R_scal(n, 0.0);
    for (size_t i = 0; i < n; i++)
    {
        R_scal[i] = 1.0;
    }

    //     //next we need to scale by a parameter theta
    double maxA = tlapack::lange(tlapack::Norm::Max, FG);

    double normA = tlapack::lange(tlapack::Norm::Inf, FG);
    reg_flops.add_double_flops(n*n);
    mixed_prec_flops.add_double_flops(n*n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            FG(i, j) = scale * FG(i, j);
        }
    }

    tlapack::lacpy(tlapack::Uplo::General, FG, A);
    for (size_t j = 0; j < n; ++j)
    {
        for (size_t i = 0; i < n; ++i)
        {
            A_F(i, j) = static_cast<T6>(FG(i, j));
            A_B(i, j) = static_cast<T2>(FG(i, j));
            A_C(i, j) = static_cast<T3>(FG(i, j));
            Zero(i, j) = 0.0;
        }
    }

    // first generate the solution vector. Since we want to reach a double precision solution, we'll generate the solution vector in double precision
    std::vector<double> x_(n);
    tlapack::LegacyVector<double, idx_t> x(n, x_.data());

    // we'll need a copy of x in each precision for calls to gemm, etc
    std::vector<double> x_1_(n);
    tlapack::LegacyVector<double, idx_t> x_1(n, x_1_.data());
    std::vector<T2> x_2_(n);
    tlapack::LegacyVector<T2, idx_t> x_2(n, x_2_.data());
    std::vector<T3> x_3_(n);
    tlapack::LegacyVector<T3, idx_t> x_3(n, x_3_.data());

    // a copy of x in float just because
    std::vector<double> x_f_(n);
    tlapack::LegacyVector<double, idx_t> x_f(n, x_f_.data());

    // b will be genrated in double and we'll store low precision copies for it just like x
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

    // bd for "true sol" -- may need to change this to quad
    // std::vector<double> bd_(n);
    // tlapack::LegacyVector<double, idx_t> bd(n, bd_.data());

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

    for (int i = 0; i < n; i++)
    {
        b[i] = static_cast<double>((-0.5 * (static_cast<double>(rand())) / static_cast<double>(RAND_MAX)));
        b[i] += static_cast<double>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)));
        b_f[i] = static_cast<double>(b[i]);
        b_1[i] = static_cast<double>(b[i]);
        b_2[i] = static_cast<T2>(b[i]);
        b_3[i] = static_cast<T3>(b[i]);
        be_1[i] = (i == 0 ? 1.0 : 0.0);
        be_1_f[i] = static_cast<T6>(be_1[i]);
        be_1_1[i] = static_cast<double>(be_1[i]);
        be_1_2[i] = static_cast<T2>(be_1[i]);
        be_1_3[i] = static_cast<T3>(be_1[i]);
        //bd[i] = static_cast<double>(b[i]);
    }

    // perform LU on A and FG
    std::vector<float> LL_(n * n);
    std::vector<int> LL_exp(n, 0);
    // tlapack::BlockMatrix<float> LU(n, n, LU_.data(), n, LU_exp.data(), n);
    tlapack::LegacyMatrix<float, idx_t> LL(n, n, LL_.data(), n);

    // keep copies in float and the second precision
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

    // first we need to get the soln in double-
    tlapack::lacpy(tlapack::GENERAL, FG, LL_double);

    //before we copy A to LL, we need to equilibrate it
    std::vector<double> D(n, 1.0);
    for(int i = 0; i < n; i++) D[i] = 1.0;
    //for(int i = n/2; i  < n; i++) D[i] = std::pow(2.0, -(int)log2(A(i,i))); 

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            A(i,j) = A(i,j)*D[i]*D[j];
        }
    }
    mixed_prec_flops.add_double_flops(n*n);

    tlapack::lacpy(tlapack::GENERAL, A, LL);



    int info = pivoted_cholesky(LL, left_piv, right_piv, chol_modif, mixed_fact_flops ,block_size, tolerance, dropping_prob);
    cout << "make high prec copy after cholesky \n";

    std::cout << "matrix after cholesky : \n";


    tlapack::lacpy(tlapack::GENERAL, LL, LL_copy);


    
 


    if (info != 0)
    {
        cout << " Cholesky in low precision failed" << endl;
        return -1.0;
    }

  

    // int infotoo = cholesky_kernel<double>(LL_double);
    // if (infotoo != 0)
    // {
    //     cout << " Matrix is not SPD" << endl;
    //     return -1.0;
    // }

    // compute true soln in double
    // tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::NO_TRANS, Diag::NonUnit, LL_double, bd);
    // tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::TRANSPOSE, Diag::NonUnit, LL_double, bd);

    //"true sol" in bd, now compute init sol with custom Cholesky

    // ++ -> transpose
    // -- -> no_trans

    //first scale b
    for(int i = 0; i < n; i++) b_3[i] *= D[i];
    mixed_prec_flops.add_float_flops(n);
    std::cout << "norm of b before pivoting\n";
    double dnormb = inf_norm(b_3);
    std::cout << dnormb << std::endl;

    for (int i = 0; i < n; i++)
    {
        auto tmp = b_3[left_piv[i]];
        b_3[left_piv[i]] = b_3[i];
        b_3[i] = tmp;
    }
    std::cout << "norm of b before trsv\n";
    dnormb = inf_norm(b_3);
    std::cout << dnormb << std::endl;


    std::cout << "norm of LL^T : "  << tlapack::lange(tlapack::Norm::Inf, LL_copy) << std::endl;
    
    
    tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::NO_TRANS, Diag::NonUnit, LL_copy, b_3);
    tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::TRANSPOSE, Diag::NonUnit, LL_copy, b_3);
    std::cout << "norm of b after trsv\n";
    dnormb = inf_norm(b_3);
    std::cout << dnormb << std::endl;

    mixed_prec_flops.add_float_flops(2*n*n);

    for (int i = n-1; i >= 0; i--)
    {
        auto tmp = b_3[right_piv[i]];
        b_3[right_piv[i]] = b_3[i];
        b_3[i] = tmp;
    }

    std::cout << "norm of b after pivoting\n";
   dnormb = inf_norm(b_3);
    std::cout << dnormb << std::endl;

    for(int i = 0; i < n ;i++) b_3[i] *= D[i];
    mixed_prec_flops.add_float_flops(n);


   

    
    for (int i = 0; i < n; i++)
        r[i] = b[i];
    for(int i = 0; i < n; i++)
        x[i] = b_3[i];
    tlapack::gemv(tlapack::NO_TRANS, -1.0, FG, x, 1.0, r); 
    mixed_prec_flops.add_double_flops(2*n*n);
    
    // Check initial backward error
    double initial_inf_norm_r = inf_norm(r);

    double norm_x = inf_norm(x);
    double initial_backward_error = initial_inf_norm_r / (norm_x * normA);
    std::cout << "initial norm_x = " << norm_x <<  std::endl;
    std::cout << "norm A = " << normA << std::endl;
    std::cout << "Initial backward error is: " << initial_backward_error << std::endl;
    std::cout << "now computing condition number for preconditioned A\n";
    

    tlapack::lacpy(tlapack::GENERAL, FG, X);

    for(int i = 0; i < n; i++) {
        auto ttmp = tlapack::col(X, left_piv[i]);
        auto ttmp2 = tlapack::col(X, i);
        tlapack::swap(ttmp, ttmp2);
        auto ttmp3 = tlapack::row(X, left_piv[i]);
        auto ttmp4 = tlapack::row(X, i);
        tlapack::swap(ttmp3, ttmp4);
    }

    

    double nrmb = tlapack::nrm2(b);
    

    
    // Initialize CG iteration counter
    int count = 0;
    

    std::vector<double> z(n); 
    std::vector<double> p(n); 
    std::vector<double> temp_p(n);
    std::vector<double> Ap(n); // A * p

    norm_x = inf_norm(x);

    // Compute initial preconditioned residual z = M^{-1} * r
    // Preconditioning steps:

    for(int i = 0; i < n; i++) {
        z[i] = r[i] * D[i];
    }
    mixed_prec_flops.add_double_flops(n);
    
    // Apply pivot P^T
    for (int i = 0; i < n; i++) {
        auto tmp = z[left_piv[i]];
        z[left_piv[i]] = z[i];
        z[i] = tmp;
    }
    
    // solve with L and L^T
    tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::NO_TRANS, tlapack::Diag::NonUnit, LL_copy, z);
    mixed_prec_flops.add_double_flops(n*n);
    

    tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::TRANSPOSE, tlapack::Diag::NonUnit, LL_copy, z);
    mixed_prec_flops.add_double_flops(n*n);

    //apply P
    for (int i = n-1; i >= 0; i--) {
        auto tmp = z[right_piv[i]];
        z[right_piv[i]] = z[i];
        z[i] = tmp;
    }
    
    // 6. Scale by D again
    for(int i = 0; i < n; i++) {
        z[i] *= D[i];
    }
    mixed_prec_flops.add_double_flops(n);
    
    // Initialize search direction p = z
    for(int i = 0; i < n; i++) {
        p[i] = z[i];
    }
    
    // Compute initial rz = r^T z
    double rz = 0.0;
    for(int i = 0; i < n; i++) {
        rz += r[i] * z[i];
    }
    mixed_prec_flops.add_double_flops(n);

    int p_iter = 0;
    for (int j = 0; j < max_CG_iter; j++) {
        //Ap = A * p
        tlapack::gemv(tlapack::NO_TRANS, 1.0, FG, p, 0.0, Ap);
        mixed_prec_flops.add_double_flops(2*n*n);
        
        // Compute p^T Ap
        double pAp = 0.0;
        for(int i = 0; i < n; i++) {
            pAp += p[i] * Ap[i];
        }
        mixed_prec_flops.add_double_flops(2*n);
        
        // Check for division by zero
        if(pAp == 0.0) {
            std::cerr << "Division by zero encountered in alpha computation." << std::endl;
            break;
        }
        
        // Compute alpha = (r^T z) / (p^T Ap)
        double alpha = rz / pAp;
        mixed_prec_flops.add_double_flops(1);
        
        // Update x = x + alpha * p
        for(int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
        }
        mixed_prec_flops.add_double_flops(n);
        
        // Update r = r - alpha * Ap
        for(int i = 0; i < n; i++) {
            r[i] -= alpha * Ap[i];
        }
        mixed_prec_flops.add_double_flops(n);
        
        // Compute norm of residual
        double inf_norm_r = inf_norm(r);
        double scaled_residual = inf_norm_r / (normA * inf_norm(x));
        mixed_prec_flops.add_double_flops(2*n + 1);
        
        // Log the residual
        myfile << j << "," << scaled_residual << std::endl;
        
        // Check for convergence
        if (scaled_residual < d_conv_bound) {
            p_iter = j+1;
            std::cout << "Converged to desired precision in " << j + 1 << " iterations." << std::endl;
            break;
        }
        
        // Precondition the new residual z = M^{-1} * r
        // Scale by D
        for(int i = 0; i < n; i++) {
            z[i] = r[i] * D[i];
        }
        mixed_prec_flops.add_double_flops(n);
        
        // Apply left permutation (P^T)
        for (int i = 0; i < n; i++) {
            auto tmp = z[left_piv[i]];
            z[left_piv[i]] = z[i];
            z[i] = tmp;
        }
        
        // Solve L y = P^T D r
        tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::NO_TRANS, tlapack::Diag::NonUnit, LL_copy, z);
        mixed_prec_flops.add_double_flops(n*n);
        
        // Solve L^T y = y
        tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::TRANSPOSE, tlapack::Diag::NonUnit, LL_copy, z);
        mixed_prec_flops.add_double_flops(n*n);
        
        // Apply right permutation (P) to y
        for (int i = n-1; i >= 0; i--) {
            auto tmp = z[right_piv[i]];
            z[right_piv[i]] = z[i];
            z[i] = tmp;
        }
        
        // Scale by D again
        for(int i = 0; i < n; i++) {
            z[i] *= D[i];
        }
        mixed_prec_flops.add_double_flops(n);
        
        // Compute new rz_new = r^T z_new
        double rz_new = 0.0;
        for(int i = 0; i < n; i++) {
            rz_new += r[i] * z[i];
        }
        mixed_prec_flops.add_double_flops(2*n);
        
        // Compute beta = rz_new / rz
        double beta = rz_new / rz;
        mixed_prec_flops.add_double_flops(1);
        
        // Update search direction p = z_new + beta * p
        for(int i = 0; i < n; i++) {
            p[i] = z[i] + beta * p[i];
        }
        mixed_prec_flops.add_double_flops(n);
        
        // Update rz for the next iteration
        rz = rz_new;
        
        // Increment iteration counter
        count++;
    }
    

    

    // Final residual computation: r = b - A * x
    for (int i = 0; i < n; i++)
        r[i] = b[i];
    tlapack::gemv(tlapack::NO_TRANS, -1.0, FG, x, 1.0, r);
    mixed_prec_flops.add_double_flops(2*n*n);
    
    // Compute final backward error
    double final_inf_norm_r = inf_norm(r);
    double final_backward_error = final_inf_norm_r / (inf_norm(x) * normA);
    std::cout << "Final backward error: " << final_backward_error << std::endl;

    std::vector<double> error(n, 0.0);
    
    // Optional: Compute norm of x
    std::cout << "Norm of x is: " << inf_norm(x) << std::endl;



    //now solve the same problem with non preconditioned CG and compare convergence
    std::cout << "======================================================================\n";
    std::cout << "now running standard CG : \n";

    for(int i = 0; i < n; i++) x[i] = 0.0;
    for(int i = 0; i < n; i++) {
        r[i] = b[i];
        p[i] = r[i];
    }

    bool convd = false;
    int v_iters = 0;

    reg_flops.add_double_flops(4*n);
    double r_norm = tlapack::dot(r, r);
    double b_norm = tlapack::dot(b, b);

       for (int iter = 1; iter <= max_CG_iter; ++iter) {
        // Compute Ap = A * p
        tlapack::gemv(tlapack::NO_TRANS, 1.0, FG, p, 0.0, Ap);
        reg_flops.add_double_flops(2*n*n);

        // Compute alpha = (r^T r) / (p^T Ap)
        double pAp = tlapack::dot(p, Ap);
        reg_flops.add_double_flops(2*n);
        if (pAp == 0.0) {
            std::cerr << "Encountered zero denominator in alpha computation." << std::endl;
            break;
        }
        double alpha = r_norm / pAp;
        reg_flops.add_double_flops(1);

        // Update x = x + alpha * p
        tlapack::axpy(alpha, p, x);
        reg_flops.add_double_flops(n);

        // Update r = r - alpha * Ap
        tlapack::axpy(-alpha, Ap, r);
        reg_flops.add_double_flops(2*n);

        // Compute new residual norm
        double r_norm_new = tlapack::dot(r, r);
        reg_flops.add_double_flops(2*n);

        // Check for convergence
        double rel_residual = std::sqrt(r_norm_new / b_norm);
        reg_flops.add_double_flops(2);
        other_file << iter << "," << rel_residual << std::endl;

        if (rel_residual < d_conv_bound) {
            v_iters = iter;
            std::cout << "Converged to double in " << iter << " iterations." << std::endl;
            break;
        }

        // Compute beta = (r_new^T r_new) / (r_old^T r_old)
        double beta = r_norm_new / r_norm;
        reg_flops.add_double_flops(1);

        // Update search direction p = r + beta * p
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        reg_flops.add_double_flops(2*n);

        // Prepare for next iteration
        r_norm = r_norm_new;
    }



    
    cout << "flops for no preconditioner CG : \n";
    reg_flops.print_stats();
    cout << "flops for Cholesky : \n";
    mixed_fact_flops.print_stats();
    std::cout << " total effective work for mixed Cholesky (normalized to fp8): \n";
    std::cout << " fraction of FLOPs done in fp8 is : " << (double)mixed_fact_flops.get_fp8_flops()/ (double) mixed_fact_flops.total_flops() << "\n";
    std::cout << mixed_fact_flops.report_total() << std::endl;
    cout << "flops for CG with precond : \n";
    mixed_prec_flops.print_stats();
    std::cout << " total effective work for precond CG (normalized to fp8): \n";
    std::cout << mixed_prec_flops.report_total() << std::endl;
    cout << "flops for fp64 Cholesky : \n";
    cholesky_flops.print_stats();


    std::cout << "stats for full mixed precision algorithm (mixed Cholesky + CG)\n";
    mixed_fact_flops.add_struct(mixed_prec_flops);
    mixed_fact_flops.print_stats();


    

    cout << " improvement in number of flops is : \n";
    std::vector<n_flops> Cholesky_flops = {mixed_fact_flops, mixed_prec_flops};
    cout << "improvemenyt over fp64 CG \n";
    cout << 1.0/reg_flops.compare_with(Cholesky_flops) << "\n";
    cout << "--------------------------------------------------\n";
    cout << "improvement over fp64 Cholesky \n";
    cout << 1.0/cholesky_flops.compare_with(Cholesky_flops) << "\n";

    std::cout << "logging results...... \n";

    string new_file = "results/n" + std::to_string(n) + "cond" + std::to_string((int)cond) + ".txt";

    std::ofstream newfile;


    newfile.open(new_file, std::ios::app);

    newfile << "eps_prime : " << tolerance << "\n";
    newfile << "Cholesky flops : \n";
    n_flops::log_all(newfile, Cholesky_flops);
    newfile << "number of iterations for precond_CG : " << p_iter << "\n";
    newfile << "CG flops : \n";
    reg_flops.log_results(newfile);
    newfile << "number of iterations for vanilla_CG : " << v_iters << "\n";
    newfile << "vanilla Cholesky flops : \n";
    cholesky_flops.log_results(newfile);



    newfile.close();

    
    return final_backward_error;
}

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
    er3 += CG_IR<float, Eigen::half, double, double, double, double, double>(n, scaling_factor, static_cast<float>(cond), fact_type, switching_val, tolerance, conv_thresh, work_factor, is_symmetric, diag_dom, dropping_prob, chol_modif, 999, 0, num_iter_1, total_num_iter, precond_kernel, max_gmres_iter, inner_gmres_num, num_IR_iter, refinement_method, block_size, stopping_pos);

    bool verified = abs(er3) < 1e-8;
    FILE *fp = fopen("./log.txt", "w");
    fputs(verified ? "true\n" : "false\n", fp);
    fprintf(fp, "%20.13E\n", static_cast<double>(er3));
    std::cout << "err3 : " << er3 << std::endl;
    return 0;
}
