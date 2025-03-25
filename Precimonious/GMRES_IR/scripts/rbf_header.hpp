#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/lapack/rscl.hpp"
#include "tlapack/lapack/getrf.hpp"
#include <fstream>
#include <queue>
#include <algorithm>

template <typename T>
int sample_index(const std::vector<T>& arr)
{
    // Compute the total sum
    T sum = std::accumulate(arr.begin(), arr.end(), T(0));
    if (sum <= T(0)) {
        return -1;
    }

    T randVal = (static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX)) * sum;

    // "Roulette wheel" selection
    T partial = T(0);
    for (int i = 0; i < static_cast<int>(arr.size()); i++) {
        partial += arr[i];
        if (partial >= randVal) {
            return i;
        }
    }
    return static_cast<int>(arr.size()) - 1;
}

template<typename T>
std::set<int> sample_indices(const std::vector<T>& A, int r) {
    std::set<int> unique_samples;
    std::vector<double> probabilities(A.size());
    double sum_A = std::accumulate(A.begin(), A.end(), 0.0);

    // Compute probabilities
    for (size_t i = 0; i < A.size(); i++) {
        probabilities[i] = A[i] / sum_A;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

    while (unique_samples.size() < static_cast<size_t>(r)) {
        int sampled_index = dist(gen) + 1; // Convert 0-based to 1-based index
        unique_samples.insert(sampled_index);
    }
    
    return unique_samples;
}

template<TLAPACK_MATRIX matrix_t>
bool is_symm(matrix_t& A) {
    
    int n = nrows(A);
    for(int i = 0; i < n; i++) {
        for(int j= 0; j  <n; j++) {
            if(A(i,j) != A(j,i)) return false;
        }
    }

    return true;

}

template <typename T>
void add_unique(std::vector<T>& vec, const T& elem) {
    if (std::find(vec.begin(), vec.end(), elem) == vec.end()) {
        vec.push_back(elem);
    }
}

template<TLAPACK_VECTOR vector_t>
int max_index(vector_t& v, int start) {
    int to_ret = start;
    int n = size(v);
    for(int i = start; i < n; i++) {
        if(abs(v[i]) > abs(v[to_ret])) to_ret = i;
    }
    return to_ret;
}
//function for regular unblocked Cholesky decomp. returns decomp in lower part of matrix
template<TLAPACK_SCALAR eps_t, TLAPACK_MATRIX matrix_t>
int cholesky_kernel(matrix_t& A, float eta = 1.0, float ksi = 1.0, int N = 1024, chol_mod chol_modif = chol_mod::NONE, bool phase2 = false, float prev_err = 0.0)
{   
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    idx_t n = nrows(A);
    real_t tmp = real_t(0.0);
    for (int i = 0; i < n; i++) {
      
    tmp = A(i,i);
    
    

    for (int k = 0; k < i; k++) {
        tmp -= A(i,k) * A(i,k); 
    }

    auto b_norm = 0.0;
    auto b_inf = 0.0;
    for(int j = i+1; j < n; j++) {
        b_norm += abs(A(j,i));
        b_inf = (b_inf > abs(A(j,i))) ? b_inf : abs(A(j, i));
    }
    // if (tmp <= 0) {
    //     cout << "Matrix is not positive definite, perturbing diagonal to restore positveness\n";
    //     tmp += 0.0625;
    //     cout << "tmp after prturbation : " << tmp << "\n";
    // }


 

    auto mach_eps = (float)std::numeric_limits<eps_t>::epsilon();
    auto NUN = (float)std::numeric_limits<eps_t>::min();
    auto DUN = (float)std::numeric_limits<eps_t>::denorm_min();
    DUN = 0.0625;
    auto beta_sq = std::max(eta, std::max(ksi/ (float)sqrt(N*N - 1), (float)mach_eps));
    // if( tmp <= 0 ) { 
    //     tmp = A(i,i) + (float)std::numeric_limits<eps_t>::epsilon(); std::cout << "negative entry! perturbing to preserve +veness";
    // } 
    auto prev_tmp = tmp;
    if(phase2 && chol_modif == chol_mod::GMW81)
    {
        tmp = abs(tmp) > DUN*2.0 ? abs(tmp) : DUN*2.0;
        tmp = tmp > b_inf*b_inf/ beta_sq ? tmp : b_inf*b_inf/ beta_sq;
        tmp = tmp > prev_err ? tmp : prev_err;
        prev_err = std::max(prev_err, tmp - prev_tmp);
    }
    else if (phase2 && chol_modif == chol_mod::SE90)
    {
        auto pert = std::max((float)0.0, std::max((float)b_norm - tmp, prev_err));
        tmp = tmp + pert;
        prev_err = pert;
    }
    else if (phase2 && chol_modif == chol_mod::SE99) 
    {
        float buf = std::max((float)b_norm, prev_err);
        auto pert = std::max((float)0.0, -tmp + buf);
        tmp = tmp + pert;
        prev_err = pert;
    }
    
    
    

    A(i,i) = sqrt((tmp));  
    

    for (int j = i+1; j < n; j++) {
        tmp = A(j,i);
        for (int k = 0; k < i; k++) {
            tmp -= A(j,k) * A(i,k);  
        }

        A(j,i) = tmp / A(i,i);  
    }


}

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < i; j++) {
            A(j,i) = A(i,j);
        }
    }


    return 0;

}



std::ofstream logfile("mgs.txt");

/// function to updated diagonal entry and check for update precision and modified cholesky phase switching
template<TLAPACK_SCALAR scalar_t, TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
bool update_diag(matrix_t& L, vector_t& updated_A, matrix_t& A, vector_t& diag_A, vector_t& masks, chol_mod chol_modif, float tau = 1.0, float gamma = 1.0, float mu = 1.0)
{
    using idx_t = size_type<matrix_t>;
    int n = nrows(L);
    int b = ncols(L);

    bool to_ret = true;
   

    for(int i = 0; i < n; i++) updated_A[i] = A(i,i);

    double to_print_sum = 0.0;

    for(int i = 0; i < n; i++) {
        to_print_sum = 0.0;
        for(int j = 0; j < b; j++) {
            updated_A[i] -= L(i,j)*L(i,j);
            to_print_sum += L(i,j)*L(i,j);
            
        }
        if(chol_modif == chol_mod::SE99) {
            to_ret = to_ret & (updated_A[i] < tau*gamma);
        } else if(chol_modif == chol_mod::SE90) {
            to_ret = to_ret & (updated_A[i] < -mu*diag_A[i] || updated_A[i] < -mu*gamma);
        } else if(chol_modif == chol_mod::GMW81) {
            to_ret = to_ret & (updated_A[i] < tau*diag_A[i]);
        }



        logfile << "to_print_sum = " << to_print_sum << "\n";
        logfile <<  "sample element = " << L(0,0) << std::endl;
        logfile << "diag we are updating = " << updated_A[i] << std::endl;
        to_print_sum = 0.0;

    }


    for(int i = 0; i < n; i++) {
        if(updated_A[i] <= 0) {masks[i] = 0.0;}
        else masks[i] = 1.0;
    }


    return to_ret;

}

template<TLAPACK_SCALAR scalar_t, TLAPACK_VECTOR vector_t, TLAPACK_MATRIX matrix_t>
bool can_use_type(matrix_t& A, vector_t& diag_A, vector_t& updated_A, double eps_prime)
{
    int n = size(diag_A);

    
    double mach_eps = (double)std::numeric_limits<scalar_t>::epsilon();
    
    for(int i = 0; i < n; i++) {     
        if (!(mach_eps*updated_A[i] < eps_prime*diag_A[i])) return false;
    }
    
    return true;   
}

template<TLAPACK_SCALAR scalar_t, TLAPACK_VECTOR vector_t, TLAPACK_MATRIX matrix_t>
bool can_use_ext_type(matrix_t& A, vector_t& diag_A, vector_t& updated_A, double eps_prime)
{
    int n = size(diag_A);

    
    double mach_eps_sq = (double)std::numeric_limits<scalar_t>::epsilon();
    mach_eps_sq *= mach_eps_sq;
    
    for(int i = 0; i < n; i++) {     
        if (!(mach_eps_sq*updated_A[i] < eps_prime*diag_A[i])) return false;
    }
    
    return true;   
}



enum precisions  : uint8_t { 
    fp32 = 0,
    fp16 = 1,
    fp8 = 2,
    fp6 = 3,
    fp4 = 4
}
