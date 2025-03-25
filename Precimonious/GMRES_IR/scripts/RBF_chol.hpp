/// randomly pivoted cholesky for RBF kernel  
/// @author: Sudhanva Kulkarni, UC berkeley
#include "rbf_header.hpp"


//function that gives slice of matrix


template<typename T>
class Compare {
public:
    bool operator()(pair<T, int> below, pair<T, int> above)
    {
        if (below.first < above.first) {
            return true;
        }
        return false;
    }
};


template <typename T>
T rbf_value(const std::vector<T>& points, int i, int j, T sigma)
{
    T diff = points[i] - points[j];
    return std::exp( - (diff * diff) / (2 * sigma * sigma) );
}

//note that the "matrix" is not initialized before hand, instead we construct it as the function progresses
template<TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR piv_t, TLAPACK_VECTOR data_t>
std::set<int> rand_piv_RBF(matrix_t& A, data_t& diag_A, std::vector<float>& points, piv_t& left_piv, piv_t& right_piv, int k,  int r = 32, double tol = 0.0000000000001, float dropping_prob = 0.1, int rank_approx = 20, float sigma = 1.0) 
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<int, int>;


    using gemm_type = float;
    using gemm_type2 =  Eigen::half;
    using gemm_type3 = lo_float::float8_e4m3fn;
    using gemm_type4 = lo_float::float8_e4m3fn;
    using gemm_type5 = lo_float::float8_e4m3fn;

    //declare precisions list
    std::vector<precisions> precs((int) k/r);

    Create<LegacyMatrix<T, idx_t>> T_matrix;
    Create<LegacyMatrix<gemm_type, idx_t>> gemm_matrix;
    Create<LegacyMatrix<gemm_type2, idx_t>> gemm_2_matrix;
    Create<LegacyMatrix<gemm_type3, idx_t>> gemm_3_matrix;
    Create<LegacyMatrix<gemm_type4, idx_t>> gemm_4_matrix;

    idx_t n = nrows(A);
    int q = n/r;
    bool phase2 = false;

    T ksi = -std::numeric_limits<T>::infinity();
    T eta = -std::numeric_limits<T>::infinity();

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(j != i) ksi = (ksi > abs(A(i,j))) ? ksi : abs(A(i,j));
        }
    }

    for(int i = 0; i < n; i++)
    {
        diag_A[i] = rbf_value(points, i, i, sigma);
    }


    using val_idx = pair<T, int>;

    if ( n <= r) return cholesky_kernel<float>(A);



    vector<T> updated_diag(n);
    for(int i = 0; i < n; i++) updated_diag[i] = diag_A[i];

    vector<T> masks(n);
    for(int i = 0; i < n; i++) masks[i] = 1.0;

    std::vector<int> sampled_columns;

    int swich = 5;
    int new_entries = 0;
    int num_curr_entries = 0;
    int num_old_entries = 0;
    std::set<int> S; // Declare S as an empty set

    //fp32 n by r buffer
    std::vector<float> fp32_buf(r*n);
    auto f_buff = gemm_matrix(fp32_buf, n, r);

    std::vector<T> F_S_(r*r);
    auto F_S = T_matrix(F_S_, r, r);

    std::vector<float> buf_L_data(r * r), buf_U_data(r * r);




    for(int i = 0; i < (int)k/r; i+= 1)
    {
        auto S_prime = sample_indices(updated_diag, r);
        S.insert(S_prime.begin(), S_prime.end());

        //now sample the cols corresponding to S_prime
        int col_idx = 0;
        for (int col : S_prime) {
            for (int row = 0; row < n; row++) {
                f_buff(row, col_idx) = rbf_value(points, row, col, sigma);
            }
            col_idx++;
        }

        auto F_S_usable = tlapack::rows(F_S, range(0, S_prime.size()));
        
        //GEMM
        for(int j = 0; j < i; j++)
        {
        switch(precs[j]) {
            case precisions::fp32 : {
                auto buf_L = gemm_matrix(buf_L_data, r, r);
                auto buf_U = gemm_matrix(buf_U_data, r, r);
                squeezing_matmul(le_F, F_S_usable, f_buff, buf_L, buf_U, -1.0, 1.0, r);
                break;
            }
            case precisions::fp16 : {
                auto buf_L = gemm_2_matrix(reinterpret_cast<gemm_type2*>(buf_L_data.data()), r, r);
                auto buf_U = gemm_2_matrix(reinterpret_cast<gemm_type2*>(buf_U_data.data()), r, r);
                squeezing_matmul(le_F, F_S_usable, f_buff, buf_L, buf_U, -1.0, 1.0, r);
                break;
            }
            case precisions::fp8 : {
                auto buf_L = gemm_3_matrix(reinterpret_cast<gemm_type3*>(buf_L_data.data()), r, r);
                auto buf_U = gemm_3_matrix(reinterpret_cast<gemm_type3*>(buf_U_data.data()), r, r);
                squeezing_matmul(le_F, F_S_usable, f_buff, buf_L, buf_U, -1.0, 1.0, r);
                break;
            }
            case precisions::fp6 : {
                auto buf_L = gemm_4_matrix(reinterpret_cast<gemm_type4*>(buf_L_data.data()), r, r);
                auto buf_U = gemm_4_matrix(reinterpret_cast<gemm_type4*>(buf_U_data.data()), r, r);
                squeezing_matmul(le_F, F_S_usable, f_buff, buf_L, buf_U, -1.0, 1.0, r);
                break;
            }
            // Add more cases if needed
            default: {
                auto buf_L = gemm_matrix(buf_L_data, r, r);
                auto buf_U = gemm_matrix(buf_U_data, r, r);
                squeezing_matmul(le_F, F_S_usable, f_buff, buf_L, buf_U, -1.0, 1.0, r);
                break;
            }
        }
    }
      

        //cholesky
        auto f_panel = tlapack::rows(f_buff, range(0, S_prime.size()));

        // Add ε_mach * tr(G) * I to stabilize the matrix before factorization
        T trace = 0;
        for (int i = 0; i < S_prime.size(); i++)
            trace += f_panel(i, i);
        T eps = std::numeric_limits<T>::epsilon();
        for (int i = 0; i < S_prime.size(); i++)
            f_panel(i, i) += eps * trace;

        cholesky_kernel(f_panel);

        //now TRSM-
        tlapack::trsm(tlapack::LEFT_SIDE, Uplo::Lower, tlapack::NO_TRANS, tlapack::Diag::NonUnit, 1.0, f_panel, f_buff);

        //update d
        for (int row = 0; row < n; row++) {
            T norm_sq = 0;
            for (int j = 0; j < S_prime.size(); j++) {
                T val = f_buff(row, j);
                norm_sq += val * val;  // assuming real type; use val * conj(val) if complex
            }
            updated_diag[row] -= norm_sq;
            updated_diag[row] = std::max(updated_diag[row], T(0)); // clip to nonnegative
        }


        

        //now I need to figure out the mixed precision stuff. I'll just do ewverythiong in fp8 for now
        // for(int i = 0; i < n; i++)
        // {
            
        // }
    
        tlapack::lacpy(tlapack::GENERAL, f_panel, A);

        precs[i] = precisions::fp32;



    }


    return S;


}

