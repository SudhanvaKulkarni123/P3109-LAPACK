///this file contains different variants of gemm with bfp
#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/rscl.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/plugins/legacyArray.hpp"

int numU_to8 = 0;
int numL_to8 = 0;

/// @brief simple gemm
template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t, TLAPACK_SCALAR scal_t>
void simple_gemm(matrixA_t& A, matrixB_t& B, matrixC_t& C, scal_t scale_by = 1.0)
{
    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            float sum = 0;
            for (idx_t l = 0; l < k; l++)
            {
                sum += scale_by * float(A(i, l)) * float(B(l, j));
            }
            C(i, j) -= sum;
        }
    }
    return;
}






/// @brief preforms C = C + AB using mocroscaling format. A gets common exp over rows and B gets common exp over columns. C is in whatever format the original matrix is in
/// @tparam matrix_t 
/// @param A 
/// @param B 
/// @param C 
template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t>
void block_gemm(matrixA_t& A, matrixA_t& B, matrixA_t& C, matrixB_t& A_dtype, matrixC_t& B_dtype)
{
    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using U = type_t<matrixC_t>;
    using dtype2 = real_type<U>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);



    
    //first find max exp in all rows for A
    std::vector<int> max_exp_A1(m, 0);
    std::vector<int> max_exp_A2(m, 0);
    for (idx_t i = 0; i < m; i++)
    {   
        if(blocks){
        int max_exp = -999;
        for (idx_t j = 0; j < k; j++)
        {
            if (int(floor(log2(abs(A(i, j))))) > max_exp) max_exp = int(floor(log2(abs(A(i, j)))));
        }
        max_exp_A1[i] = 0;
        } else {
            max_exp_A1[i] = 0;
        }
    }

    //now find max exp in all columns for B
    std::vector<int> max_exp_B1(n, 0);
    for (idx_t j = 0; j < n; j++)
    {   
        if(blocks){
        int max_exp = -999;
        for (idx_t i = 0; i < k; i++)
        {
            if (int(floor(log2(abs(B(i, j))))) > max_exp) max_exp = int(floor(log2(abs(B(i, j)))));
        }
        max_exp_B1[j] = 0;
        } else {
            max_exp_B1[j] = 0;
        }
    }

    //now scale A and B accordingly -- A is lower triangular, B upper
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < k; j++)
        {
            A_dtype(i, j) = static_cast<dtype>((A(i, j)/std::pow(2.0,max_exp_A1[i])));
        }
    }

    for (idx_t i = 0; i < k; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            B_dtype(i, j) = static_cast<dtype2>(B(i, j)/ std::pow(2.0,max_exp_B1[j]));
        }
    }

    //now perform the matmul as would be done in tensor cores
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            float sum = 0;
            for (idx_t l = 0; l < k; l++)
            {
                auto first_leftover  = static_cast<dtype>(std::pow(2,7)*(static_cast<float>(A(i,l)) - std::pow(2.0,max_exp_A1[i])*static_cast<float>(A_dtype(i,l))));
                auto second_leftover = static_cast<dtype2>(std::pow(2,7)*(static_cast<float>(B(l,j)) - std::pow(2.0,max_exp_B1[j])*static_cast<float>(B_dtype(l,j))));
                //sum += static_cast<float>(static_cast<float>(std::pow(2,7))*static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j)) + (static_cast<float>((first_leftover)) * static_cast<float>(B_dtype(l, j)) + static_cast<float>(A_dtype(i, l)) * static_cast<float>((second_leftover))))/std::pow(2, 7);
                // + std::pow(2,-14)*(static_cast<float>(first_leftover) * static_cast<float>(second_leftover));
                sum += static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j));
            }
            C(i, j) -= sum*std::pow(2.0,max_exp_A1[i] + max_exp_B1[j]);
        }
    }
    isNanorInf(C);
    // Free the vectors max_exp_A and max_exp_B



    return;

}



/// @brief preforms C = C + AB using mocroscaling format. A and B get common exp over a 4-by-4 block 
/// @tparam matrix_t
/// @param A
/// @param B
/// @param C
template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t>
void fbfmatmul(matrixA_t& A, matrixA_t& B, matrixA_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, int block_size = 2)
{

    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using range = std::pair<idx_t, idx_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);


    for(int i = 0; i < m/block_size; i++) 
    {
        for(int j = 0; j < n/block_size; j++) 
        {
            auto C_p = tlapack::slice(C, range(i*block_size, (i+1)*block_size), range(j*block_size, (j+1)*block_size));
            for(int l = 0; l < k/block_size; l++){

                auto A_p = tlapack::slice(A, range(i*block_size, (i+1)*block_size), range(l*block_size, (l+1)*block_size));
                auto B_p = tlapack::slice(B, range(l*block_size, (l+1)*block_size), range(j*block_size, (j+1)*block_size));
                
                auto Adp = tlapack::slice(A_dtype, range(0, (1)*block_size), range(0, (1)*block_size));
                auto Bdp = tlapack::slice(B_dtype, range(0, block_size), range(0, block_size));
                //take exp common
                int max_exp_A = -999;
                int max_exp_B = -999;
                for(int ii = 0; ii < block_size; ii++){
                    for(int jj = 0; jj < block_size; jj++) {
                        if (int(floor(log2(abs(A_p(ii, jj))))) > max_exp_A) max_exp_A = int((log2(abs(A_p(ii, jj)))));
                        if (int(floor(log2(abs(B_p(ii, jj))))) > max_exp_B) max_exp_B = int((log2(abs(B_p(ii, jj)))));
                    }
                }
                max_exp_A -= 2;
                max_exp_B -= 2;


                for(int ii = 0; ii < block_size; ii++){
                    for(int jj = 0; jj < block_size; jj++) {
                        Adp(ii, jj) = static_cast<dtype>(A_p(ii, jj)/std::pow(2.0,max_exp_A));
                        Bdp(ii, jj) = static_cast<dtype>(B_p(ii, jj)/std::pow(2.0,max_exp_B));
                    }
                }
                simple_gemm(Adp, Bdp, C_p, std::pow(2.0,max_exp_A + max_exp_B));

            }
            
        }
    }


    return;


}


/// @brief preforms C = C + AB using mocroscaling format. updates on diagonal are done in fp16, off-diagonal in fp8
template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t>
void fbfmatmul_fp16(matrixA_t& A, matrixA_t& B, matrixA_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, matrixC_t& o_A_dtype, matrixC_t& o_B_dtype, int block_size = 4)
{

    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using range = std::pair<idx_t, idx_t>;
    using odtype = real_type<type_t<matrixC_t>>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);




    
    //first find max exp in all rows for A
    std::vector<int> max_exp_A1(m, 0);
    std::vector<int> max_exp_A2(m, 0);
    for (idx_t i = 0; i < m; i++)
    {   
        if(blocks){
        int max_exp = -999;
        for (idx_t j = 0; j < k; j++)
        {
            if (int(floor(log2(abs(A(i, j))))) > max_exp) max_exp = int(floor(log2(abs(A(i, j)))));
        }
        max_exp_A1[i] = 0;
        } else {
            max_exp_A1[i] = 0;
        }
    }

    //now find max exp in all columns for B
    std::vector<int> max_exp_B1(n, 0);
    for (idx_t j = 0; j < n; j++)
    {   
        if(blocks){
        int max_exp = -999;
        for (idx_t i = 0; i < k; i++)
        {
            if (int(floor(log2(abs(B(i, j))))) > max_exp) max_exp = int(floor(log2(abs(B(i, j)))));
        }
        max_exp_B1[j] = 0;
        } else {
            max_exp_B1[j] = 0;
        }
    }

    //now scale A and B accordingly -- A is lower triangular, B upper
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < k; j++)
        {
            A_dtype(i, j) = static_cast<dtype>((A(i, j)/std::pow(2.0,max_exp_A1[i])));
        }
    }
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < k; j++)
        {
            o_A_dtype(i, j) = static_cast<odtype>((A(i, j)/std::pow(2.0,max_exp_A1[i])));
        }
    }

    for (idx_t i = 0; i < k; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            B_dtype(i, j) = static_cast<dtype>(B(i, j)/ std::pow(2.0,max_exp_B1[j]));
        }
    }
    for (idx_t i = 0; i < k; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            o_B_dtype(i, j) = static_cast<odtype>(B(i, j)/ std::pow(2.0,max_exp_B1[j]));
        }
    }



    //now perform the matmul as would be done in tensor cores
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            float sum = 0;
            for (idx_t l = 0; l < k; l++)
            {
                auto first_leftover  = static_cast<dtype>(std::pow(2,7)*(static_cast<float>(A(i,l)) - std::pow(2.0,max_exp_A1[i])*static_cast<float>(A_dtype(i,l))));
                auto second_leftover = static_cast<dtype>(std::pow(2,7)*(static_cast<float>(B(l,j)) - std::pow(2.0,max_exp_B1[j])*static_cast<float>(B_dtype(l,j))));
                
                if(i >= j-4 && i <= j+4){ 
                    sum += static_cast<float>(o_A_dtype(i, l)) * static_cast<float>(o_B_dtype(l, j));
                } else {
                    //sum += static_cast<float>(static_cast<float>(std::pow(2,7))*static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j)) + (static_cast<float>((first_leftover)) * static_cast<float>(B_dtype(l, j)) + static_cast<float>(A_dtype(i, l)) * static_cast<float>((second_leftover))))/std::pow(2, 7);
                    // + std::pow(2,-14)*(static_cast<float>(first_leftover) * static_cast<float>(second_leftover));
                    sum += static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j));
                }
            }
            C(i, j) -= sum*std::pow(2.0,max_exp_A1[i] + max_exp_B1[j]);
        }
    }
    isNanorInf(C);
    // Free the vectors max_exp_A and max_exp_B





    return;
}




template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t>
void squeezing_matmul(matrixA_t& A, matrixA_t& B, matrixC_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, float z1, float z2, int block_size = 4)
{

    using idx_t = size_type<matrixA_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;
    using V = type_t<matrixB_t>;
    using dtype = real_type<V>;
    using range = std::pair<idx_t, idx_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(B);
    const idx_t k = ncols(A);

    //find signed max and min of A, B
    float max_A = -999;
    float min_A = 999;
    float max_B = -999;
    float min_B = 999;
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < k; j++)
        {
            if ((A(i, j)) > max_A) max_A = (A(i, j));
            if ((A(i, j)) < min_A) min_A = (A(i, j));
        }
    }

    //find signed max and min in B
    for (idx_t i = 0; i < k; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            if ((B(i, j)) > max_B) max_B = (B(i, j));
            if ((B(i, j)) < min_B) min_B = (B(i, j));
        }
    }

    auto alpha1 = (z2 - z1)/(max_A - min_A);
    auto alpha2 = (z2 - z1)/(max_B - min_B);

    auto beta1 = (z1*max_A - z2*min_A)/(max_A - min_A);
    auto beta2 = (z1*max_B - z2*min_B)/(max_B - min_B);

    // squueze into A_dtype and B_dtype

    std::vector<float> A_sums(m, 0);
    for (idx_t i = 0; i < m; i++)
    {
        for (idx_t j = 0; j < k; j++)
        {
            A_dtype(i, j) = static_cast<dtype>(alpha1*A(i, j) + beta1);
            A_sums[i] += alpha1*A(i, j) + beta1;
        }
    }

    std::vector<float> B_sums(n, 0);

    for (idx_t i = 0; i < k; i++)
    {
        for (idx_t j = 0; j < n; j++)
        {
            B_dtype(i, j) = static_cast<dtype>(alpha2*B(i, j) + beta2);
            B_sums[j] += alpha2*B(i, j) + beta2;
        }
    }

    for(int i = 0; i < m; i++) 
    {
        for(int j = 0; j < n; j++) 
        {
            float sum = 0;
            for(int l = 0; l < k; l++)
            {
                sum += (static_cast<float>(A_dtype(i, l)) * static_cast<float>(B_dtype(l, j)));
            }
            sum -= beta2*A_sums[i] + beta1*B_sums[j] - beta1*beta2*k;
            sum = sum/(alpha1*alpha2);
            C(i, j) -= sum;
        }
    }


    return;

    

    



}


template<TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixB_t, TLAPACK_MATRIX matrixC_t>
void multi_block_gemm(matrixA_t& A, matrixA_t& B, matrixA_t& C, matrixB_t& A_dtype, matrixB_t& B_dtype, matrixC_t& A_dtype2, matrixC_t& B_dtype2, float normA = 1.0, double eps = 1.0/256.0, int block_size = 4)
{
    
        using idx_t = size_type<matrixA_t>;
        using T = type_t<matrixA_t>;
        using real_t = real_type<T>;
        using V = type_t<matrixB_t>;
        using dtype = real_type<V>;
        using W = type_t<matrixC_t>;
        using dtype2 = real_type<W>;
        using range = std::pair<idx_t, idx_t>;
        const idx_t m = nrows(A);
        const idx_t n = ncols(B);
        const idx_t k = ncols(A);
        bool A_to_8 = false;
        bool B_to_8 = false;

        for(int i = 0; i < m/4; i++) 
        {
            for(int j = 0; j < n/4 ; j++)
            {
                auto C_p = tlapack::slice(C, range(i*4, (i+1)*4), range(j*4, (j+1)*4));
                for(int l = 0; l < k/4; l++)
                {
                    auto A_p = tlapack::slice(A, range(i*4, (i+1)*4), range(l*4, (l+1)*4));
                    auto B_p = tlapack::slice(B, range(l*4, (l+1)*4), range(j*4, (j+1)*4));
                    A_to_8 = tlapack::lange(tlapack::Norm::Fro, A_p) < 32.0*eps*normA;
                    B_to_8 = tlapack::lange(tlapack::Norm::Fro, B_p) < 32.0*eps*normA;
                    for(int ii = 0; ii < 4; ii++)
                    {
                        for(int jj = 0; jj < 4; jj++)
                        {   
                            if(A_to_8) A_dtype(ii, jj) = static_cast<dtype>(A_p(ii, jj));
                            else A_dtype2(ii, jj) = static_cast<dtype2>(A_p(ii, jj));

                            if(B_to_8) B_dtype(ii, jj) = static_cast<dtype>(B_p(ii, jj));
                            else B_dtype2(ii, jj) = static_cast<dtype2>(B_p(ii, jj));
                        }
                    }

                    if(A_to_8 && B_to_8) {
                        numL_to8++;
                        numU_to8++;
                        simple_gemm(A_dtype, B_dtype, C_p, 1.0);
                    } else if(A_to_8 && !B_to_8) {
                        numL_to8++;
                        simple_gemm(A_dtype, B_dtype2, C_p, 1.0);
                    } else if(!A_to_8 && B_to_8) {
                        numU_to8++;
                        simple_gemm(A_dtype2, B_dtype, C_p, 1.0);
                    } else {
                        simple_gemm(A_dtype2, B_dtype2, C_p, 1.0);
                    }
                    

                }
            }
        }
        return;
}