/// This file contains functions to calculate the weighted opcount of BLAS calls in GMRES-IR
/// @author Sudhanva Kulkarni UC Berkeley


template<typename T>
double adjust_weight(double unweighted)
{
    return unweighted;
}

template<>
double adjust_weight<float>(double unweighted)
{
    return unweighted * 4.0;
}


template<>
double adjust_weight<double>(double unweighted)
{
    return unweighted * 8.0;
}

template<>
double adjust_weight<Eigen::bfloat16>(double unweighted)
{
    return unweighted * 2.0;
}


template<>
double adjust_weight<Eigen::half>(double unweighted)
{
    return unweighted * 2.0;
}

template<>
double adjust_weight<ml_dtypes::float8_ieee_p<4>>(double unweighted)
{
    return unweighted;
}


template <typename matrix_At>
double adjust_weight(double unweighted, matrix_At& A)
{
    using TA = type_t<matrix_At>;
    return adjust_weight<TA>(unweighted);

}

template<typename matrix_At, typename matrix_Bt, typename vector_xt, typename vector_bt>
double get_base_opcount(const char* funcname, const matrix_At& A, const matrix_Bt& B, const vector_xt& x, const vector_bt& b, int num_iter)
{
    auto A_rows = adjust_weight(nrows(A), A);
    auto A_cols = adjust_weight(ncols(A), A);
    auto B_rows = adjust_weight(nrows(B), B);
    auto B_cols = adjust_weight(ncols(B), B);
    auto x_size = adjust_weight(size(x), x);
    auto b_size = adjust_weight(size(b), b);
    {
        if(funcname == "gemv") return 2.0 * A_rows * A_cols + 2.0 * A_rows;
        else if(funcname == "trsv") return A_rows * A_cols;
        else if(funcname == "trsm") return B_cols * A_rows * A_cols;
        else if(funcname == "gemm") return 2.0 * B_rows * A_cols * A_rows;
        else if(funcname == "trmm") return B_rows * A_rows * A_cols;
        else if(funcname == "Hessenberg QR") return 3.0 * A_rows * (3.0 + A_rows);
        else if(funcname == "Arnoldi_iter") return (2.0 * A_rows * A_cols) + 2.0*A_rows + double(num_iter + 1)*(2*A_rows - 1) + 2.0 * A_rows*( num_iter + 1.0);
        else if(funcname == "getrf") return 2.0 * A_rows * A_cols;
        else if(funcname == "axpy") return 2.0 * x_size;
        else if(funcname == "dot") return 2.0 * x_size;
        else if(funcname == "nrm2") return 2.0 * x_size + 1.0;
        else if(funcname == "scal") return x_size;
        else if(funcname == "rscl") return x_size;
        else return 0.0;
    }
}

template<typename matrix_At, typename matrix_Bt, typename vector_xt, typename vector_bt>
double get_adjusted_opcount(const char* funcname, const matrix_At& A, const matrix_Bt& B, const  vector_xt& x, const vector_bt& b, int num_iter)
{
    return get_base_opcount(funcname, A, B, x, b, num_iter);
    
}
