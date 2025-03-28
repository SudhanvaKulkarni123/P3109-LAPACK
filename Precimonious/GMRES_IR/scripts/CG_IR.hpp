std::ofstream myfile("e5m2_error_f_cond.csv");
std::ofstream other_file("e5m2_error_e_cond.csv");
std::ofstream timefile("time.txt");

template <typename Matrix, typename Vector, typename T3, typename T6, bool verbose = false>
double CG_IR(Matrix& FG, Vector& b, int n, double tolerance, double conv_thresh, chol_mod chol_modif, int block_size = 128)
{

    using idx_t = size_t;
    using range = pair<idx_t, idx_t>;



    // auto TR = lo_float::float6_p<3>(1.0);

    double d_conv_bound = sqrt(static_cast<double>(n)) * std::pow(2, -32);
    double s_conv_bound = sqrt(static_cast<double>(n)) * std::pow(2, -24);

    
        n_flops mixed_prec_flops(); mixed_prec_flops.reset();
        //n_flops reg_flops(); reg_flops.reset();
        n_flops mixed_fact_flops();  mixed_fact_flops.reset();
        //n_flops cholesky_flops(); cholesky_flops.reset();

        //cholesky flops is just n^3/3
        //auto nn = (long) n;
        //cholesky_flops.add_double_flops(nn*nn*(nn/3));
    


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

   
        // reg_flops.add_double_flops(n*n);
        mixed_prec_flops.add_double_flops(n*n);
    



    tlapack::lacpy(tlapack::Uplo::General, FG, A);

    std::vector<double> x_(n);
    tlapack::LegacyVector<double, idx_t> x(n, x_.data());


    // Generate b in T3
    std::vector<T3> b_3_(n);
    tlapack::LegacyVector<T3, idx_t> b_3(n, b_3_.data());


    std::vector<double> r_(n);
    tlapack::LegacyVector<double, idx_t> r(n, r_.data());


    for (int i = 0; i < n; i++){
    
        b_3[i] = static_cast<T3>(b[i]);

    }



    // keep copies in float and the second precision
    std::vector<T6> LL_copy_(n * n);
    tlapack::LegacyMatrix<T6, idx_t> LL_copy(n, n, LL_copy_.data(), n);


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


    int info = pivoted_cholesky(A, left_piv, right_piv, chol_modif, mixed_fact_flops ,block_size, tolerance);



    tlapack::lacpy(tlapack::GENERAL, A, LL_copy);


    
 


    if (info != 0)
    {
        cout << " Cholesky in low precision failed" << endl;
        return -1.0;
    }

  

    //first scale b
    for(int i = 0; i < n; i++) b_3[i] *= D[i];
    mixed_prec_flops.add_float_flops(n);


    for (int i = 0; i < n; i++)
    {
        auto tmp = b_3[left_piv[i]];
        b_3[left_piv[i]] = b_3[i];
        b_3[i] = tmp;
    }

    
    
    tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::NO_TRANS, Diag::NonUnit, LL_copy, b_3);
    tlapack::trsv(tlapack::LOWER_TRIANGLE, tlapack::TRANSPOSE, Diag::NonUnit, LL_copy, b_3);


    mixed_prec_flops.add_float_flops(2*n*n);

    for (int i = n-1; i >= 0; i--)
    {
        auto tmp = b_3[right_piv[i]];
        b_3[right_piv[i]] = b_3[i];
        b_3[i] = tmp;
    }

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
    if constexpr (verbose)
    {
        std::cout << "initial norm_x = " << norm_x <<  std::endl;
        std::cout << "norm A = " << normA << std::endl;
        std::cout << "Initial backward error is: " << initial_backward_error << std::endl;
        std::cout << "now computing condition number for preconditioned A\n";
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
        if (scaled_residual < conv_thresh) {
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

    return final_backward_error;
}


    
template <typename Matrix, typename Vector, typename T3, typename T6, bool verbose = false>
void vanilla_CG(Matrix& FG, Vector& b, int n, double tolerance, double conv_thresh, chol_mod chol_modif, int block_size = 128)
{

    using idx_t = size_t;
    using range = pair<idx_t, idx_t>;
    using T = type_t<Matrix>;
    using real_t = real_type<T>;

    
    //logging data structures-
    //n_flops mixed_prec_flops(); mixed_prec_flops.reset();
    n_flops reg_flops(); reg_flops.reset();
    //n_flops mixed_fact_flops();  mixed_fact_flops.reset();
    //n_flops cholesky_flops(); cholesky_flops.reset();


    //declare x and other vectors just as in precond function
    std::vector<double> x_(n);
    tlapack::LegacyVector<double, idx_t> x(n, x_.data());

    std::vector<double> r_(n);
    tlapack::LegacyVector<double, idx_t> r(n, r_.data());


    std::vector<double> z(n); 
    std::vector<double> p(n); 
    std::vector<double> temp_p(n);
    std::vector<double> Ap(n); // A * p




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

        if (rel_residual < conv_thresh) {
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

    
    return;
}
