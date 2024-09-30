/// @author Sudhanva Kulkarni, University of California Berkeley, USA
/// This file contains some relevant versions of the GMRES-IR routine.


enum class refinement_type
{
    GMRES_IR,
    NVIDIA_IR,
    NO_IR,
};
enum class kernel_type
{
    LEFT_LU,
    RIGHT_LU,
    RIGHT_GMRES,
    SPLIT_LU,
    NONE
};

enum class ortho_type
{
    MGS,
    DHH,
    RHH
};

int num_higher_prec_gmres_loops = 0;
int num_lower_prec_gmres_loops = 0;
int buffer_int = 0;

std::ofstream FGMRES_log("FGMRES_log.txt");

template<typename matrixA_t, typename matrixQ_t, typename matrixH_t, typename matrixLU_t, typename vector_t, typename idk>
class GMRES_args
{
public:
    int m;
    int max_num_iter;
    double tol;
    std::vector<double> err;
    vector_t cs;
    vector_t sn;
    int inner_num_gmres;
    matrixA_t A;
    vector_t b;
    vector_t x0;
    vector_t x;
    vector_t r;
    std::vector<idk> piv;
    kernel_type ker;
    int info;
    GMRES_args(matrixA_t& A, matrixQ_t& Q, matrixH_t& H, matrixLU_t& LU, std::vector<idk>& piv, vector_t& b, vector_t& x0, vector_t& x, vector_t& r, vector_t& cs, vector_t& sn, kernel_type ker, int m, int max_num_iter, double tol = 1e-8, int inner_num_gmres = 20) : A(A), Q(Q), H(H), LU(LU), piv(piv), b(b), x0(x0), x(x), r(r), cs(cs), sn(sn), ker(ker), m(m), max_num_iter(max_num_iter), tol(tol), inner_num_gmres(inner_num_gmres)
    {
        tlapack::lacpy(tlapack::GENERAL, A, LU);
        info = tlapack::getrf(LU, piv);
    }

private:
    matrixLU_t LU;
    matrixQ_t Q;
    matrixH_t H;


};

template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    for(int i = 0; i < nrows(A); i++)
    {
        for(int j = 0; j < ncols(A); j++)
        {
            std::cout << A(i,j) << " ";
        }
        std::cout << std::endl;
    }

}



template <typename T>
void rotmat(T a, T b, T& c, T& s) 
{
if ( b == T(0.0) ){
    c = 1.0;
    s = 0.0;
}
else if ( abs(b) > abs(a) ){
    auto temp = a / b;
    auto temp2 = temp*temp; 
    s = T(1.0) /  (sqrt(( T(1.0) + temp2 )));
    c = temp * s;
}
else {
    auto temp = b / a;
    auto temp2 = temp*temp;
    c = T(1.0) / (sqrt(( T(1.0) + temp2 )));
    s = temp * c;
}

return;
}


template <typename matrix_t, typename vector_t>
void compute_residual(matrix_t &A, vector_t &x, vector_t& b, vector_t& r, float scale = 1.0)
{
    // data traits
    using idx_t = size_type<matrix_t>;
    using scalar_t = type_t<matrix_t>;
    using range = pair<idx_t, idx_t>;
    int n = size(r);
    for (int i = 0; i < n; i++)
        r[i] = b[i];

    // compute r = b - A*x
    tlapack::gemv(Op::NoTrans, static_cast<scalar_t>(-1.0), A, x, static_cast<scalar_t>(1.0), r);
    

    return;
}

template <typename vector_t>
double inf_norm(vector_t &v)
{
    double max = 0;
    for (int i = 0; i < size(v); i++)
    {
        if (abs(double(v[i])) > max)
        {
            max = abs(double(v[i]));
        }
    }
    return max;
}

/// solution is returned in b
template <typename matrix_t, typename vector_t, typename idk>
void LU_solve(matrix_t &LU, std::vector<idk> &piv, vector_t &b, bool is_symmetric = false)
{
    // data traits
    using idx_t = size_type<matrix_t>;
    using scalar_t = type_t<matrix_t>;
    using range = pair<idx_t, idx_t>;

    int n = size(b);

    for (int i = 0; i < n; i++)
    {
        if (piv[i] != i)
        {
            auto tmp = b[piv[i]];
            b[piv[i]] = b[i];
            b[i] = tmp;
        }
    }
    if(is_symmetric) {
        tlapack::trsv(Uplo::Lower, tlapack::NO_TRANS, Diag::NonUnit, LU, b);
        tlapack::trsv(Uplo::Lower, tlapack::TRANSPOSE, Diag::NonUnit, LU, b);

    } else {
        tlapack::trsv(Uplo::Lower, tlapack::NO_TRANS, Diag::Unit, LU, b);
        tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, Diag::NonUnit, LU, b);
    }
    
    return;
}

// this function will convert H into an upper triangular R and b into Q^Tb. Then we can solve Rx = Q^Tb outside this function
template <typename matrix_t, typename vector_t>
void Hessenberg_qr(matrix_t H, vector_t b, int size)
{
    // data traits
    using idx_t = size_type<matrix_t>;
    using real_t = type_t<matrix_t>;
    using range = pair<idx_t, idx_t>;

    auto m = nrows(H);
    auto n = ncols(H);
    real_t c = 0.0;
    real_t s = 0.0;
    real_t temp = 0.0;

    auto da_num = n < size ? n : size - 1;
    for (int i = 0; i < da_num; i++)
    {
        c = H(i, i);
        s = -H(i + 1, i);
        H(i, i) = sqrt(H(i, i) * H(i, i) + H(i + 1, i) * H(i + 1, i));
        c = c / H(i, i);
        s = s / H(i, i);
        H(i + 1, i) = 0.0;
        for (int j = i + 1; j < n; j++)
        {
            temp = c * H(i, j) - s * H(i + 1, j);
            H(i + 1, j) = s * H(i, j) + c * H(i + 1, j);
            H(i, j) = temp;
        }
        temp = c * b[i] - s * b[i + 1];
        b[i + 1] = s * b[i] + c * b[i + 1];
        b[i] = temp;
    }

    return;
}

// this function will performs the i-th iteration of Hessenberg QR. It will update H 
template <typename matrix_t, typename vector_t, typename other_vec_t>
void Updating_Hessenberg_qr(matrix_t& H, vector_t& be_1, int size, int i, other_vec_t cs, other_vec_t sn)
{
    //k = i here
    // data traits
    using idx_t = size_type<matrix_t>;
    using real_t = type_t<matrix_t>;
    using range = pair<idx_t, idx_t>;

    auto m = nrows(H);
    auto n = ncols(H);
    real_t c = 0.0;
    real_t s = 0.0;
    
  
    //6*(k-1)

    for(int k = 0; k <= i-1; k++) {
            auto temp     =  cs[k]*H(k,i) + sn[k]*H(k+1,i);
            H(k+1,i) = -sn[k]*H(k,i) + cs[k]*H(k+1,i);
            H(k,i)   = temp;
        }
        rotmat(H(i,i), H(i+1,i), cs[i], sn[i]);
        auto temp   = cs[i]*be_1[i];                       
        be_1[i+1] = -sn[i]*be_1[i];
        be_1[i]   = temp;
        H(i,i) = cs[i]*H(i,i) + sn[i]*H(i+1,i);
        H(i+1,i) = 0.0;
        std::cout << "cs[i] is : " << cs[i] << " sn[i] is : " << sn[i] << std::endl;

    return;


    
}

/// this function perform one step of arnoldi iter using regular gram schmidt
template <TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixH_t, TLAPACK_MATRIX matrixQ_t, TLAPACK_MATRIX matrixLU_t, typename idk>
void arnoldi_iter(matrixA_t &A, matrixH_t &H, matrixQ_t &Q, matrixLU_t LU, std::vector<idk> piv, kernel_type ker, int k)
{
    // data traits
    using idx_t = size_type<matrixA_t>;
    using scalar_t = type_t<matrixA_t>;
    using range = pair<idx_t, idx_t>;
    using prod_type = type_t<matrixLU_t>;

    // constants
    const idx_t n = nrows(Q);
    const idx_t m = n;

    std::vector<prod_type> Q_tmp_(n * n);
    tlapack::LegacyMatrix<prod_type, idx_t> Q_tmp(n, n, Q_tmp_.data(), n);
    tlapack::lacpy(tlapack::GENERAL, Q, Q_tmp);
    std::vector<prod_type> A_tmp_(n * n);
    tlapack::LegacyMatrix<prod_type, idx_t> A_tmp(n, n, A_tmp_.data(), n);
    tlapack::lacpy(tlapack::GENERAL, A, A_tmp);

    // temporary storage
    std::vector<scalar_t> w(n);
    std::vector<prod_type> w1(n);
    std::vector<prod_type> w2(n);

    // one step of Arnoldi iteration
    // w = A * V[j]
    if (ker == kernel_type::LEFT_LU)
    {
        auto vec_tmp = tlapack::col(Q_tmp, k);
        gemv(Op::NoTrans, static_cast<prod_type>(1.0), A_tmp, vec_tmp, static_cast<prod_type>(0), w1);
        // need to permute before applying LU
        for (int i = 0; i < n; i++)
        {
            if (piv[i] != i)
            {
                auto tmp = w1[piv[i]];
                w1[piv[i]] = w1[i];
                w1[i] = tmp;
            }
        }
        tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU, w1);
        tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU, w1);
        

        for (int i = 0; i < n; i++)
            w[i] = static_cast<scalar_t>(w1[i]);
    }
    else if (ker == kernel_type::RIGHT_LU)
    {

        // invert then apply A
        auto vec_tmp = tlapack::col(Q_tmp, k);
        for (int i = 0; i < n; i++)
            w1[i] = static_cast<prod_type>(vec_tmp[i]);
        for (int i = 0; i < n; i++)
        {
            if (piv[i] != i)
            {
                auto tmp = w1[piv[i]];
                w1[piv[i]] = w1[i];
                w1[i] = tmp;
            }
        }
        tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU, w1);
        tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU, w1);

        tlapack::gemv(Op::NoTrans, static_cast<prod_type>(1.0), A_tmp, w1, static_cast<prod_type>(0), w2);
        for (int i = 0; i < n; i++)
            w[i] = static_cast<scalar_t>(w2[i]);
    } else if (ker == kernel_type::SPLIT_LU)
    {
        // left precond by PL and right by U
        auto vec_tmp = tlapack::col(Q_tmp, k);
        for (int i = 0; i < n; i++)
            w1[i] = static_cast<prod_type>(vec_tmp[i]);
        
        
        tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU, w1);

        tlapack::gemv(Op::NoTrans, static_cast<prod_type>(1.0), A_tmp, w1, static_cast<prod_type>(0), w2);
        for (int i = 0; i < n; i++)
            w[i] = static_cast<scalar_t>(w2[i]);

        for (int i = 0; i < n; i++)
        {
            if (piv[i] != i)
            {
                auto tmp = w[piv[i]];
                w[piv[i]] = w[i];
                w[i] = tmp;
            }
        }
        tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU, w);

    }
    else if (ker == kernel_type::NONE)
    {
        auto vec_tmp = tlapack::col(Q_tmp, k);
        for (int i = 0; i < n; i++)
            w1[i] = static_cast<prod_type>(vec_tmp[i]);
        tlapack::gemv(Op::NoTrans, static_cast<prod_type>(1.0), A_tmp, w1, static_cast<prod_type>(0), w2);
        for (int i = 0; i < n; i++)
            w[i] = static_cast<scalar_t>(w2[i]);
    }

    // H[j,0:j+1] = V[0:n] * w
    for (idx_t i = 0; i <= k+1; i++) {
        H(i, k) = dot(tlapack::col(Q, i), w);     // 2n
        axpy(-H(i, k), tlapack::col(Q, i), w);    // 2n
    }

       

    if (k == n - 1)
        return;
    // H[k+1,k] = ||w||
    H(k + 1, k) = nrm2(w);          //2n

    if (H(k + 1, k) == scalar_t(0))
        std::cout << "breakdown occured" << std::endl;

    // Q[k+1] = w / H[k+1,k]

    rscl(H(k + 1, k), w);       // n
    for (int i = 0; i < n; i++)
    {
        Q(i, k+1) = w[i];
    }

    return;
}

/// This function uses randomized HHQR for arnoldi
template <typename matrixA_t, typename matrixH_t, typename matrixQ_t, typename matrixLU_t, typename vector_t, typename idk>
void rand_arnoldi(matrixA_t &A, matrixH_t &H, matrixQ_t &Q, matrixLU_t LU, std::vector<idk> piv, kernel_type ker, int k)
{   

    return;
}

/// this solves Ax = b by GMRES with initial guess x0. Workspace arrays Q,H and r must be provided by the user. Solution is returned in x0
template <typename matrixA_t, typename matrixQ_t, typename matrixH_t, typename matrixLU_t, typename vector_t, typename aux_vector_t, typename idk>
void GMRES(matrixA_t& A, matrixQ_t& Q, matrixH_t& H, matrixLU_t& LU, std::vector<idk>& piv, vector_t& b, vector_t& x0, vector_t& x, aux_vector_t& cs, aux_vector_t& sn, kernel_type ker, int m, int max_num_iter, double tol = 1e-6)
{

    using idx_t = size_type<matrixA_t>;
    using scalar_t = type_t<matrixH_t>;
    using range = pair<idx_t, idx_t>;
    auto n = size(x);
    int i = 0;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j <n; j++) {
            H(i,j) = 0.0;
            Q(i,j) = 0.0;
        }
    }

    std::vector<scalar_t> be_1(n);
    if (ker == kernel_type::LEFT_LU)
    {
        LU_solve(LU, piv, b);
    } else if(ker == kernel_type::SPLIT_LU)
    {
        for (int i = 0; i < n; i++)
    {
        if (piv[i] != i)
        {
            auto tmp = b[piv[i]];
            b[piv[i]] = b[i];
            b[i] = tmp;
        }
    }
    tlapack::trsv(Uplo::Lower, tlapack::NO_TRANS, Diag::Unit, LU, b);
    }
    //flops.add_double_flops(2*n*n);
    auto norm_r = tlapack::nrm2(b);
    //flops.add_double_flops(2*n);
    for (i = 0; i < n; i++)
    {
        Q(i, 0) = b[i] / norm_r;
    }
    //flops.add_double_flops(n);
    double prev_err = 0.0;
    for(int j = 0; j < n; j++) be_1[j] = (j == 0 ? static_cast<scalar_t>(norm_r) : static_cast<scalar_t>(0.0));
    for (i = 0; i < n - 1; i++)
    {   
        num_higher_prec_gmres_loops++;
        arnoldi_iter(A, H, Q, LU, piv, ker, i);
        //flops.add_double_flops(2*n*n);
        for(int k = 0; k <= i-1; k++) {
            auto temp     =  cs[k]*H(k,i) + sn[k]*H(k+1,i);
            H(k+1,i) = -sn[k]*H(k,i) + cs[k]*H(k+1,i);
            H(k,i)   = temp;
        }

        rotmat(H(i,i), H(i+1,i), cs[i], sn[i]);
        auto temp   = cs[i]*be_1[i];                       
        be_1[i+1] = -sn[i]*be_1[i];
        be_1[i]   = temp;
        H(i,i) = cs[i]*H(i,i) + sn[i]*H(i+1,i);
        H(i+1,i) = 0.0;
            //flops.add_double_flops(6*(i-1));
        if(abs(static_cast<double>(be_1[i+1])/static_cast<double>(norm_r)) <= tol){ 
            FGMRES_log << "converged and broke out at iteration : " << i << std::endl; 
            auto da_tmp = tlapack::slice(be_1,range{0, i});

            tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS,tlapack::Diag::NonUnit ,tlapack::slice(H,range{0, i}, range{0,i}), da_tmp);

            //flops.add_double_flops(2*(i-1)*(i-1));
            tlapack::gemv(tlapack::NO_TRANS, static_cast<scalar_t>(1.0), tlapack::slice(Q, range{0, n}, range{0,i}), da_tmp, static_cast<scalar_t>(0.0), x);
            //flops.add_double_flops(2*n*(i-1));
            break;}
        if(abs(static_cast<double>(be_1[i])/static_cast<double>(norm_r)) <= 0.125*prev_err){ FGMRES_log << "Did not make satisfactory progress and broke out at iteration : " << i << std::endl; break;}
        prev_err = abs(static_cast<double>(be_1[i])/static_cast<double>(norm_r));
        //flops.add_double_flops(1);

    }

    auto da_tmp = tlapack::slice(be_1,range{0, i});

    tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS,tlapack::Diag::NonUnit ,tlapack::slice(H,range{0, i}, range{0,i}), da_tmp);

    //flops.add_double_flops(2*(i-1)*(i-1));
    tlapack::gemv(tlapack::NO_TRANS, static_cast<scalar_t>(1.0), tlapack::slice(Q, range{0, n}, range{0,i}), da_tmp, static_cast<scalar_t>(0.0), x);
    


    if(ker == kernel_type::RIGHT_LU)
    {
        LU_solve(LU, piv, x);
    } else if(ker == kernel_type::SPLIT_LU)
    {
        tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, tlapack::Diag::NonUnit, LU, x);
    }
    //flops.add_double_flops(2*n*n);

    // Remove the unnecessary call to clear() since be_1 is a local variable and will be automatically destroyed when it goes out of scope.


    return ;
    

    

    
}

template <typename matrixA_t, typename matrixQ_t, typename matrixH_t, typename matrixLU_t, typename vector_t, typename aux_vector_t, typename idk>
void carson_GMRES(matrixA_t& A, matrixQ_t& Q, matrixH_t& H, matrixLU_t& LU, std::vector<idk>& piv, vector_t& b, vector_t& x0, vector_t& x, aux_vector_t& cs, aux_vector_t& sn, kernel_type ker, int m, int max_num_iter, double tol = 1e-6)
{
    using idx_t = size_type<matrixA_t>;
    using scalar_t = type_t<matrixH_t>;
    using range = pair<idx_t, idx_t>;
    auto n = size(x);
    int i = 0;
    for(i = 0; i < n; i++) {
        for(int j = 0; j <n; j++) {
            H(i,j) = 0.0;
            Q(i,j) = 0.0;
        }
    }

    LU_solve(LU, piv, b);

    auto nrmb = tlapack::nrm2(b);

    std::cout << "norm b is : " << nrmb << std::endl;

    cout << "U(n,n) is : " << LU(n-1, n-1) << endl;

    std::vector<scalar_t> be_1(n);
    std::vector<scalar_t> w(n);
    for(i = 0; i < n; i++) {
        Q(i, 0) = b[i]/nrmb;
        if(i == 0) be_1[i] = nrmb;
        else be_1[i] = 0.0;
    }
    int num_iter = 0;
    double prev_err = 9999999.0;

    for(i = 0; i < n-1; i++)
    {
        num_iter++;
        auto vcur = tlapack::col(Q, i);
        tlapack::gemv(tlapack::NO_TRANS, static_cast<scalar_t>(1.0), A, vcur, static_cast<scalar_t>(0.0), w);
        // need to permute before applying LU
        for (int f = 0; f < n; f++)
        {
            if (piv[f] != f)
            {
                auto tmp = w[piv[f]];
                w[piv[f]] = w[f];
                w[f] = tmp;
            }
        }
        tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU, w);
        tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU, w);

        for(int k = 0; k <= i; k++) {
            H(k, i) = tlapack::dot(tlapack::col(Q, k), w);
            tlapack::axpy(-H(k, i), tlapack::col(Q, k), w);
        }
        H(i+1, i) = tlapack::nrm2(w);
        for(int k = 0; k < n; k++) {
            Q(k, i+1) = w[k]/H(i+1, i);
        }
        for(int k = 0; k <= i-1; k++) {
            auto temp     =  cs[k]*H(k,i) + sn[k]*H(k+1,i);
            H(k+1,i) = -sn[k]*H(k,i) + cs[k]*H(k+1,i);
            H(k,i)   = temp;
        }
        rotmat(H(i,i), H(i+1,i), cs[i], sn[i]);
        auto temp   = cs[i]*be_1[i];                       
        be_1[i+1] = -sn[i]*be_1[i];
        be_1[i]   = temp;
        H(i,i) = cs[i]*H(i,i) + sn[i]*H(i+1,i);
        H(i+1,i) = 0.0;

        

        
        if(i == n-1 || abs(be_1[i+1])/nrmb <= tol) {
            std::cout << "converged and broke out at iteration : " << i << std::endl;
            auto tmp = tlapack::slice(be_1, range{0, i});
            tlapack::trsv(tlapack::Uplo::Upper, tlapack::NO_TRANS, tlapack::Diag::NonUnit, tlapack::slice(H, range{0, i}, range{0, i}),tmp );
            tlapack::gemv(tlapack::NO_TRANS, 1.0, tlapack::slice(Q, range{0, n}, range{0, i}), tmp, 0.0, x);
            break;
        } 
       

        if(abs(static_cast<double>(be_1[i+1])/static_cast<double>(nrmb)) >= prev_err){ cout << "Did not make satisfactory progress and broke out at iteration : " << i << std::endl; break;}


        prev_err = abs(be_1[i+1])/nrmb;


    }

    num_higher_prec_gmres_loops += i+1;

    // auto tmp = tlapack::slice(be_1, range{0, i});
    //         tlapack::trsv(tlapack::Uplo::Upper, tlapack::NO_TRANS, tlapack::Diag::NonUnit, tlapack::slice(H, range{0, i}, range{0, i}),tmp );
    //         tlapack::gemv(tlapack::NO_TRANS, 1.0, tlapack::slice(Q, range{0, n}, range{0, i}), tlapack::slice(be_1, range{0, i}), 0.0, x);
    // return;
    return;






}

template <TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixA_B_t, TLAPACK_MATRIX matrixH_t, TLAPACK_MATRIX matrixH_B_t, TLAPACK_MATRIX matrixQ_t, TLAPACK_MATRIX matrixQ_B_t, TLAPACK_MATRIX matrixZ_t, TLAPACK_MATRIX matrixLU_t, TLAPACK_MATRIX matrixLU_B_t, TLAPACK_VECTOR vector_t, typename idk>
void flexible_arnoldi_iter(matrixA_t &A, matrixA_B_t& A_B, matrixH_t &H, matrixH_B_t &H_B, matrixQ_t &Q, matrixQ_B_t& Q_B, matrixZ_t& Z, matrixLU_t& LU, matrixLU_B_t& LU_B, std::vector<idk>& piv, vector_t& cs, vector_t& sn, kernel_type ker, int k, int num_inner_gmres_iter, int num_inner_IR_iter)
{
    // data traits
    using idx_t = size_type<matrixA_t>;
    using scalar_t = type_t<matrixA_t>;
    using b_type = type_t<matrixA_B_t>;
    using range = pair<idx_t, idx_t>;
    using prod_type = type_t<matrixLU_t>;

    if(ker == kernel_type::RIGHT_LU || ker == kernel_type::LEFT_LU)
    {
        arnoldi_iter(A, H, Q, LU, piv, ker, k);
        return;
    }
    else if(ker == kernel_type::RIGHT_GMRES)
    {
        // constants
    const idx_t n = nrows(Q);
    const idx_t m = n;

    std::vector<prod_type> Q_tmp_(n * n);
    tlapack::LegacyMatrix<prod_type, idx_t> Q_tmp(n, n, Q_tmp_.data(), n);
    tlapack::lacpy(tlapack::GENERAL, Q, Q_tmp);
    std::vector<prod_type> A_tmp_(n * n);
    tlapack::LegacyMatrix<prod_type, idx_t> A_tmp(n, n, A_tmp_.data(), n);
    tlapack::lacpy(tlapack::GENERAL, A, A_tmp);

    std::vector<prod_type> Q_work_(n*n);
    tlapack::LegacyMatrix<prod_type, idx_t> Q_work(n, n, Q_work_.data(), n);

    std::vector<prod_type> H_work_(n*n);
    tlapack::LegacyMatrix<prod_type, idx_t> H_work(n, n, H_work_.data(), n);

    // temporary storage
    std::vector<scalar_t> w(n);
    std::vector<b_type> w1(n);
    std::vector<double> w2(n);
    std::vector<prod_type> w1_prod(n);
    std::vector<prod_type> w2_prod(n);
    std::vector<b_type> x(n);
    std::vector<b_type> r(n);
    std::vector<b_type> r1(n);
    std::vector<prod_type> r_prod(n);
    int count = 0;

    auto vec_tmp = slice(Q_tmp, range{0, m}, k);
    for (int i = 0; i < n; i++){
        w1_prod[i] = static_cast<prod_type>(vec_tmp[i]);
        w2_prod[i] = static_cast<prod_type>(vec_tmp[i]);
    }

    LU_solve(LU, piv, w2_prod);

 
    //want to call GMRES here instead of LU -- but can call LU for initial guess
     while(count < num_inner_IR_iter)
    {
    compute_residual(A, w2_prod, w1_prod, r_prod);
    if(inf_norm(r_prod)/(inf_norm(x)*tlapack::lange(tlapack::Norm::Inf, A)) <= 1e-5) break;
    FGMRES_log << " L2 norm of residual in flexible arnoldi is : " << tlapack::nrm2(r) << std::endl;
    FGMRES_log << "L-inf norm of residual in flexible arnoldi is : " << inf_norm(r) << std::endl;
    
    for(int i = 0; i < n; i++) { r[i] = static_cast<b_type>(r_prod[i]); r1[i] = r[i]; }
    LU_solve(LU_B, piv, r1);
    for(int i = 0; i < n; i++) { w2[i] = static_cast<double>(w2_prod[i]); w1[i] = static_cast<b_type>(w1_prod[i]);}
    GMRES(A_B, Q_B, H_B, LU_B, piv, r, r1, x, cs, sn, kernel_type::LEFT_LU, num_inner_gmres_iter, 6, 1e-5);
    for(int i = 0; i < n; i++) {
        w2[i] = static_cast<double>(x[i]) + (w2[i]);
    }
    for(int i = 0; i < n; i++) w2_prod[i] = static_cast<prod_type>(w2[i]);
    count++;
    }
    for(int i = 0; i < n; i++)
    {
        Z(i,k) = static_cast<scalar_t>(w2[i]);
    }
    for(int i = 0; i < n; i++) w1_prod[i] = static_cast<prod_type>(w1[i]);
    for(int i = 0; i < n; i++) w2_prod[i] = static_cast<prod_type>(w2[i]);
    
    

    tlapack::gemv(Op::NoTrans, static_cast<prod_type>(1.0), A_tmp, w2_prod, static_cast<prod_type>(0), w1_prod);
    for (int i = 0; i < n; i++)
        w[i] = static_cast<scalar_t>(w1_prod[i]);
    
  
    for (idx_t i = 0; i < k + 1; ++i){
        H(i, k) = dot(slice(Q, range{0, m}, i), w);
        axpy(-H(i, k), slice(Q, range{0, m}, i), w);
    }

    // w = w - V[0:n] * H[0:j+1,j]
        

    if (k == n - 1)
        return;
    // H[k+1,k] = ||w||
    H(k + 1, k) = nrm2(w);

    if (H(k + 1, k) == scalar_t(0))
        std::cout << "breakdown occured" << std::endl;

    // Q[k+1] = w / H[k+1,k]

    rscl(H(k + 1, k), w);
    for (int i = 0; i < m; i++)
    {
        Q(i, k + 1) = w[i];
    }
    }

    
}

template <typename matrixA_t, typename matrixA_B_t, typename matrixQ_t, typename matrixQ_B_t, typename matrixH_t, typename matrixH_B_t, typename matrixZ_t, typename vectorb_t, typename vectorx_t, typename vector_t, typename matrixLU_t, typename matrixLU_B_t, typename vector_B_t, typename idk>
void FGMRES(matrixA_t &A, matrixA_B_t& A_B, matrixQ_t &Q, matrixQ_B_t& Q_B, matrixH_t &H, matrixH_B_t& H_B, matrixZ_t& Z, matrixLU_t &LU, matrixLU_B_t& LU_B, std::vector<idk>& piv, vectorb_t &b, vectorx_t &x0, vectorx_t& x, vectorx_t &r, vector_B_t& fcs, vector_B_t& fsn, vector_t& cs, vector_t& sn, kernel_type ker, int m, int max_num_iter, double tol = 1e-8, int inner_num_gmres = 20, int num_IR_iter = 5)
{

    using idx_t = size_type<matrixA_t>;
    using scalar_t = type_t<matrixA_t>;
    using range = pair<idx_t, idx_t>;
    using b_type = type_t<matrixA_B_t>;
    auto n = size(r);
    int i = 0;

    for(int i = 0; i < n; i++) {
        for(int j = 0; j <n; j++) {
            H(i,j) = 0.0;
            Q(i,j) = 0.0;
            A_B(i,j) = b_type(ml_dtypes::float8_ieee_p<4>(A_B(i,j)));
        }
    }
    std::vector<scalar_t> be_1(n);
        if (ker == kernel_type::LEFT_LU)
        {
            LU_solve(LU, piv, b);
        }
        auto norm_r = tlapack::nrm2(b);
        for (i = 0; i < n; i++)
        {
            Q(i, 0) = b[i] / norm_r;
        }
        for(int j = 0; j < n; j++) be_1[j] = (j == 0 ? norm_r : 0.0);
        auto da_tmp = tlapack::slice(be_1,range{0, i-1});

        double prev_err = 0.0;
        for (i = 1; i < m; i++)
        {
            num_higher_prec_gmres_loops++;
            int old_num = num_higher_prec_gmres_loops;
            flexible_arnoldi_iter(A, A_B, H, H_B, Q, Q_B, Z, LU, LU_B, piv, cs, sn, ker, i-1, inner_num_gmres, num_IR_iter);
            num_lower_prec_gmres_loops += num_higher_prec_gmres_loops - old_num;
            num_higher_prec_gmres_loops = old_num;
            Updating_Hessenberg_qr(H, be_1, i, i-1, fcs, fsn);
            if(abs(static_cast<double>(be_1[i])/static_cast<double>(norm_r)) <= tol){ FGMRES_log << "converged and broke out at iteration : " << i << std::endl; break;}
            if(abs(static_cast<double>(be_1[i])/static_cast<double>(norm_r)) <= 0.125*prev_err){ FGMRES_log << "Did not make satisfactory progress and broke out at iteration : " << i << std::endl; break;}
            prev_err = abs(static_cast<double>(be_1[i])/static_cast<double>(norm_r));

        }
        tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS,tlapack::Diag::NonUnit ,tlapack::slice(H,range{0, i-1}, range{0,i-1}), da_tmp); 
        if(ker == kernel_type::RIGHT_GMRES)
        {
        tlapack::gemv(tlapack::NO_TRANS, static_cast<scalar_t>(1.0), tlapack::slice(Z, range{0, n}, range{0, i-1}), da_tmp, static_cast<scalar_t>(0.0), x);
        } else {
        tlapack::gemv(tlapack::NO_TRANS, static_cast<scalar_t>(1.0), tlapack::slice(Q, range{0, n}, range{0, i-1}), da_tmp, static_cast<scalar_t>(0.0), x);
        }
        if(ker == kernel_type::RIGHT_LU)
        {
            LU_solve(LU, piv, x);
        }


    
}

