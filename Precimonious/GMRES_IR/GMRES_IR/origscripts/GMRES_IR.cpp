///@author Sudhanva Kulkarni, UC Berkeley
//this file contains code for testing GMRES-IR using 8-bit LU decomposition as a preconditioner
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/float8_iee_p.hpp>
#include <tlapack/plugins/eigen_bfloat.hpp>
#include <tlapack/plugins/eigen_half.hpp>
#include "tlapack.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>


// template <typename matrix_t>
// void printMatrix(const matrix_t& A)
// {
//     using idx_t = tlapack::size_type<matrix_t>;
//     const idx_t m = tlapack::nrows(A);
//     const idx_t n = tlapack::ncols(A);

//     for (idx_t i = 0; i < m; ++i) {
//         std::cout << std::endl;
//         for (idx_t j = 0; j < n; ++j)
//             std::cout << A(i, j) << " ";
//     }
// }
//------------------------------------------------------------------------------
template<typename T>
std::vector<T> convertPythonListToVector(std::vector<T>& vec,PyObject* pyList) {

    if (!PyList_Check(pyList)) return vec;

    Py_ssize_t size = PyList_Size(pyList);
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item = PyList_GetItem(pyList, i);
        vec.push_back(static_cast<T>(PyFloat_AsDouble(item)));
    }

    return vec;
}

//-------------------------------------------------------------------

//this function will convert H into an upper triangular R and b into Q^Tb. Then we can solve Rx = Q^Tb outside this function
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

    auto da_num = n < size ? n : size-1;
    for(int i = 0; i < da_num; i++) {
        c = H(i,i);
        s = -H(i+1,i);
        H(i,i) = sqrt(H(i,i)*H(i,i) + H(i+1,i)*H(i+1,i));
        c = c/H(i,i);
        s = s/H(i,i);
        H(i+1,i) = 0.0;
        for(int j = i+1; j < n; j++) {
            temp = c*H(i,j) - s*H(i+1,j);
            H(i+1,j) = s*H(i,j) + c*H(i+1,j);
            H(i,j) = temp;
        }
        temp = c*b[i] - s*b[i+1];
        b[i+1] = s*b[i] + c*b[i+1];
        b[i] = temp;
        
    }
    

}







//------------------------------------------------------------------------------
//this function performs step k of arnoldi iter where k > 0
template <TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixH_t, TLAPACK_MATRIX matrixQ_t>
void arnoldi_iter(matrixA_t& A, matrixH_t& H, matrixQ_t& Q, matrixA_t LU, int k)
{
    // data traits
    using idx_t = size_type<matrixA_t>;
    using scalar_t = type_t<matrixA_t>;
    using range = pair<idx_t, idx_t>;
    // constants
    const idx_t n = nrows(Q);
    const idx_t m = ncols(H);

    // temporary storage
    std::vector<scalar_t> w(n);

    // one step of Arnoldi iteration
    // w = A * V[j]
        auto vec = slice(Q, range{0, m} ,k);
        gemv(Op::NoTrans, static_cast<scalar_t>(1.0), A, vec, static_cast<scalar_t>(0), w);
        tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU, w);
        tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU, w);

        // H[j,0:j+1] = V[0:n] * w
        for (idx_t i = 0; i < k+1; ++i)
            H(i, k) = dot(slice(Q, range{0, m} ,i), w);

        // w = w - V[0:n] * H[0:j+1,j]
        for (idx_t i = 0; i < k+1; ++i)
            axpy(-H(i, k), slice(Q, range{0, m} ,i), w);
            
        if(k == n-1) return;
        // H[k+1,k] = ||w||
        H(k+1, k) = nrm2(w);

        // Q[k+1] = w / H[k+1,k]
       
        rscl(H(k+1, k), w);
        for(int i = 0; i < m; i++) {
            Q(i,k+1) = w[i];
        }
        
    
}










//------------------------------------------------------------------------------

template<typename T, typename matrix_t>
int constructMatrix(int n, float cond, int space, bool geom, matrix_t& A, int p, float& true_cond) {
    //this is an ambitious function that uses a Python embedding to call the functions found in generate\ copy.py to fill in the entries of A
    
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
    int i;
    setenv("PYTHONPATH", ".", 1);
    Py_Initialize();
    pName = PyUnicode_DecodeFSDefault((char*)"gen");
    pModule = PyImport_Import(pName); 
    Py_DECREF(pName);
    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, (char *)"LU_gen");

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(5);
            for (i = 0; i < 5; ++i) {
                switch(i) {
                    case 0:
                        pValue = PyLong_FromLong(n);
                        break;
                    case 1:
                        pValue = PyFloat_FromDouble(cond);
                        break;
                    case 2:
                        pValue = PyLong_FromLong(space);
                        break;
                    case 3:
                        pValue = geom ? Py_True : Py_False;
                        break;
                    default:
                        pValue = PyLong_FromLong(p);
                        break;
                }
                
                if (!pValue) {
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
                    fprintf(stderr, "Cannot convert argument\n");
                    return 1;
                }
                /* pValue reference stolen here: */
                PyTuple_SetItem(pArgs, i, pValue);
            }
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                std::vector<T> b(n*n);
                std::vector<T> c(n*n + 1);
                for(int i = 0 ; i < n; i++) {
                    for(int j = 0; j < n; j++){
                        A(i,j) = static_cast<float>(static_cast<T>(PyFloat_AsDouble(PyList_GetItem(pValue, n*i + j))));
                    }
                }

                
                // tlapack::LegacyMatrix<T, int> LU(n, n, b.data(), n);
                // printMatrix(LU);
                true_cond = PyFloat_AsDouble(PyList_GetItem(pValue, n*n));
                std::cout << true_cond << std::endl;
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function for gen\n");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load program\n");
        return 1;
    }
    if (Py_FinalizeEx() < 0) {
        return 120;
    }
    return 0;
    }




std::ofstream myfile("e5m2_error_f_cond.csv");

//each T_i is such that T_i+1 is a higher precision than T_i (so T_i+1 > T_i)
template <typename T1, typename T2, typename T3>
float run(size_t n, T1 scale, float cond, int p, int variant = 0, int num_iter_1 = 0, int num_total_iter = 5) 
{   
    using matrix_t = tlapack::LegacyMatrix<T1>;
    using real_t = tlapack::real_type<T1>;
    using idx_t = size_t;
    using range = pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Create the n-by-n matrix A in precision T1
    std::vector<T1> A_(n * n);
    tlapack::LegacyMatrix<T1, idx_t> A(n, n, A_.data(), n);

    // FG matrix is "for generation" so it will be stored in double precision
    std::vector<double> FG_(n * n);
    tlapack::LegacyMatrix<double, idx_t> FG(n, n, FG_.data(), n);

    //we'll also need a single precision copy of A
    std::vector<float> A_float_(n * n);
    tlapack::LegacyMatrix<float, idx_t> A_float(n, n, A_float_.data(), n);

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
    constructMatrix<T1>(n, cond, std::ceil(cond/static_cast<float>(5)) > n-1 ? n-1 : std::ceil(cond/static_cast<float>(5)) , true, FG, p, true_cond);
   
    // std::vector<float> S_(n*n, 0.0);
    // tlapack::LegacyMatrix<float, idx_t> S(n, n, S_.data(), n);
    // for (size_t i = 0; i < n; i++){
    //     S(i, i) = 1.0;
    // }
    // std::vector<float> R_(n*n, 0.0);
    // tlapack::LegacyMatrix<float, idx_t> R(n, n, R_.data(), n);
    // for (size_t i = 0; i < n; i++){
    //     R(i, i) = 1.0;
    // }

    // float maxR, maxS;
    // std::vector<float> AR(n), AS(n);

    //  int not_count = 0;
    // while(true){
    //     for(int i = 0; i < n; i++){
    //         auto c1 = tlapack::rows(FG,range(i,i+1));
    //         AR[i] = 1/sqrt(tlapack::lange(tlapack::Norm::One, c1));
    //         auto c2 = tlapack::cols(FG, range(i,i+1));
    //         AS[i] = 1/sqrt(tlapack::lange(tlapack::Norm::One, c2));
    //         maxR = AR[i] > maxR ? AR[i] : maxR;
    //         maxS = AS[i] > maxS ? AS[i] : maxS;

    //     }
    //     for(int j = 0; j < n; j++){
    //         for(int k = 0; k < n; k++){
    //             FG(j,k) = FG(j,k)*(AR[j])*(AS[k]);
    //         }
    //     }
    //     for(int i = 0 ; i < n; i++){
    //         R(i,i) = R(i,i)*AR[i];
    //         S(i,i) = S(i,i)*AS[i];
    //     }
    //     //std::cout << maxR;
    //     not_count++;
    //     if(abs(maxR - 1) < 1 || abs(maxS - 1) < 1 || not_count > 10) break;
    //     }

    //     //next we need to scale by a parameter theta
        double maxA = tlapack::lange(tlapack::Norm::Max, FG);

        
        double normA = tlapack::lange(tlapack::Norm::Inf, FG);

    //first we'll get cond aafter preconditioning
    
    for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < n; ++i){
            //A(i,j) = static_cast<real_t>(sqrt(float(scale)*0.125)*FG(i,j)/normA);
            A(i,j) = static_cast<T1>(FG(i,j));
            A_float(i,j) = static_cast<float>(FG(i,j));
            A_B(i,j) = static_cast<T2>(FG(i,j));
            A_C(i,j) = static_cast<T3>(FG(i,j));
            Zero(i,j) = 0.0;
        }
     }

    //now generate all vectors

    //first generate the solution vector. Since we want to reach a double precision solution, we'll generate the solution vector in double precision
    std::vector<double> x_(n);
    tlapack::LegacyVector<double, idx_t> x(n, x_.data());

    //we'll need a copy of x in each precision for calls to gemm, etc
    std::vector<T1> x_1_(n);
    tlapack::LegacyVector<T1, idx_t> x_1(n, x_1_.data());
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

    std::vector<T1> b_1_(n);
    tlapack::LegacyVector<T1, idx_t> b_1(n, b_1_.data());

    std::vector<T2> b_2_(n);
    tlapack::LegacyVector<T2, idx_t> b_2(n, b_2_.data());

    // Generate b in T3
    std::vector<T3> b_3_(n);
    tlapack::LegacyVector<T3, idx_t> b_3(n, b_3_.data());

    // Generate b in float
    std::vector<float> b_f_(n);
    tlapack::LegacyVector<float, idx_t> b_f(n, b_f_.data());


    //bd for "true sol" -- may need to change this to quad
    std::vector<double> bd_(n);
    tlapack::LegacyVector<double, idx_t> bd(n, bd_.data());

    std::vector<double> r_(n);
    tlapack::LegacyVector<double, idx_t> r(n, r_.data());

    std::vector<T1> r_1_(n);
    tlapack::LegacyVector<T1, idx_t> r_1(n, r_1_.data());

    std::vector<T2> r_2_(n);
    tlapack::LegacyVector<T2, idx_t> r_2(n, r_2_.data());

    std::vector<T3> r_3_(n);
    tlapack::LegacyVector<T3, idx_t> r_3(n, r_3_.data());

    std::vector<float> r_f_(n);
    tlapack::LegacyVector<float, idx_t> r_f(n, r_f_.data());

    std::vector<double> solved_r_(n);
    tlapack::LegacyVector<double, idx_t> solved_r(n, solved_r_.data());

    std::vector<T1> solved_r_1_(n);
    tlapack::LegacyVector<T1, idx_t> solved_r_1(n, solved_r_1_.data());

    std::vector<T2> solved_r_2_(n);
    tlapack::LegacyVector<T2, idx_t> solved_r_2(n, solved_r_2_.data());

    std::vector<T3> solved_r_3_(n);
    tlapack::LegacyVector<T3, idx_t> solved_r_3(n, solved_r_3_.data());

    std::vector<float> solved_r_f_(n);
    tlapack::LegacyVector<float, idx_t> solved_r_f(n, solved_r_f_.data());



    std::vector<double> be_1_(n);
    tlapack::LegacyVector<double, idx_t> be_1(n, be_1_.data());

    std::vector<T1> be_1_1_(n);
    tlapack::LegacyVector<T1, idx_t> be_1_1(n, be_1_1_.data());

    std::vector<T2> be_1_2_(n);
    tlapack::LegacyVector<T2, idx_t> be_1_2(n, be_1_2_.data());

    std::vector<T2> be_1_3_(n);
    tlapack::LegacyVector<T2, idx_t> be_1_3(n, be_1_3_.data());

    
    std::vector<float> be_1_f_(n);
    tlapack::LegacyVector<float, idx_t> be_1_f(n, be_1_f_.data());


    for( int i = 0; i < n; i++) {
        b[i] = static_cast<double>(static_cast<real_t>(-0.5*(static_cast<double>(rand()))*double(scale)/static_cast<double>(RAND_MAX)));
        b[i] += static_cast<double>(static_cast<real_t>(static_cast<double>(rand())*double(scale)/static_cast<double>(RAND_MAX)));
        b_f[i] = static_cast<float>(b[i]);
        b_1[i] = static_cast<T1>(b[i]);
        b_2[i] = static_cast<T2>(b[i]);
        b_3[i] = static_cast<T3>(b[i]);
        be_1[i] = (i == 0 ? 1.0 : 0.0);
        be_1_f[i] = static_cast<float>(be_1[i]);
        be_1_1[i] = static_cast<T1>(be_1[i]);
        be_1_2[i] = static_cast<T2>(be_1[i]);
        be_1_3[i] = static_cast<T3>(be_1[i]);
    }

    //perform LU on A and FG
    std::vector<T1> LU_(n * n);
    tlapack::LegacyMatrix<T1, idx_t> LU(n, n, LU_.data(), n);

    //keep copies in float and the second precision
    std::vector<float> LU_copy_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_copy(n, n, LU_copy_.data(), n);

    std::vector<T2> LU_copy_B_(n * n);
    tlapack::LegacyMatrix<T2, idx_t> LU_copy_B(n, n, LU_copy_B_.data(), n);

    std::vector<T3> LU_copy_C_(n * n);
    tlapack::LegacyMatrix<T3, idx_t> LU_copy_C(n, n, LU_copy_C_.data(), n);

    std::vector<double> LU_double_(n * n);
    tlapack::LegacyMatrix<double, idx_t> LU_double(n, n, LU_double_.data(), n);

    //we need H is float and T2
    std::vector<float> H_(n*n);
    tlapack::LegacyMatrix<float, idx_t> H(n, n, H_.data(), n);

    std::vector<T2> H_B_(n*n);
    tlapack::LegacyMatrix<T2, idx_t> H_B(n, n, H_B_.data(), n);


    std::vector<float> H_copy_(n*n);
    tlapack::LegacyMatrix<float, idx_t> H_copy(n, n, H_copy_.data(), n);

    std::vector<T2> H_copy_B_(n*n);
    tlapack::LegacyMatrix<T2, idx_t> H_copy_B(n, n, H_copy_B_.data(), n);

    std::vector<float> Q_(n*n);
    tlapack::LegacyMatrix<float, idx_t> Q(n, n, Q_.data(), n);

    std::vector<T2> Q_B_(n*n);
    tlapack::LegacyMatrix<T2, idx_t> Q_B(n, n, Q_B_.data(), n);


   
    // Matrix A is kept unchanged
    tlapack::lacpy(tlapack::GENERAL, A, LU);
    tlapack::lacpy(tlapack::GENERAL, FG, LU_double);
   
    //declare arrays for piv
    std::vector<size_t> piv_lo(n);
    std::vector<size_t> piv_hi(n);

    int info, infotoo;
    if(variant == 0)    info = tlapack::getrf(LU, piv_lo, tlapack::GetrfOpts{GetrfVariant::Recursive});
    else info = tlapack::getrf(LU, piv_lo, tlapack::GetrfOpts{GetrfVariant::Level0});
    if (info != 0) {
        std::cerr << "Matrix could not be factorized :(" << std::endl;
        return -1;
    }
    int infotoo2 = tlapack::getrf(LU_double, piv_hi);
    if (infotoo2 != 0) {
        std::cerr << "Matrix could not be factorized in fp64 :(" << std::endl;
        return -1;
    }

    //compute sol in double

    for (idx_t i = 0; i < n;i++){
        if (piv_hi[i] != i) {
            auto tmp = bd[piv_hi[i]];
            bd[piv_hi[i]] = bd[i];
            bd[i] = tmp;
        }
    }


    //now "true sol" is in bd. But actually how do we define "true sol" if we want to reach double precision? Do i need to use quad??? How did higham and pranesh do it?
    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_double, bd);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_double, bd);

    for (idx_t i = 0; i < n;i++){
        if (piv_lo[i] != i) {
            auto tmp = b_3[piv_lo[i]];
            b_3[piv_lo[i]] = b_3[i];
            b_3[i] = tmp;
        }
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            LU_copy(i,j) = static_cast<float>(LU(i,j));
            LU_copy_B(i,j) = static_cast<T2>(LU(i,j));
            LU_copy_C(i,j) = static_cast<T3>(LU(i,j));
        }
    }


    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_copy_C, b_3);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_copy_C, b_3);
    //init soln is now in b_3
    

    
    //now we can begin the actual IR
    double res_norm = 1.0;
    float inner_res_norm = 1.0;

    int count = 0; //to keep track of number of IR iterations
    T2 normb_B = static_cast<T2>(0.0);

    int num_iter = 0; //number of iters for GMRES
    float tol = std::pow(10,-6);
    for(int i = 0; i < n; i++) x[i] = static_cast<double>(b_3[i]);
    for(int i = 0; i < n; i++) x_2[i] = static_cast<T2>(b_3[i]);

    //this is the first iteration of IR that uses a lower precision
    while(count < num_iter_1) { 
        count = count + 1;
        for(int i = 0; i < n; i++) 
        {
            r[i] = static_cast<double>(b_3[i]);
            r_2[i] = static_cast<T2>(b_3[i]);
        }
        //find residual
        tlapack::gemv(NO_TRANS, (-1.0), FG, x, (1.0), r);
        for(int i = 0; i < n; i++) r_2[i] = static_cast<T2>(r[i]);
        T2 m;
        res_norm = 0.0;
        for(int i = 0; i < n; i++) {
            m = m > static_cast<T2>(abs(bd[i])) ? m : static_cast<T2>(abs(bd[i]));
            res_norm = res_norm > static_cast<double>(abs(r_2[i])) ? res_norm : static_cast<double>(abs(r_2[i]));
        }

        res_norm = res_norm/(tlapack::lange(tlapack::INF_NORM, FG)*static_cast<double>(m));  
        myfile << count << "," << res_norm << std::endl;
        
        //copy r to solved_r
        for(int i = 0; i < n; i++) {
            solved_r_2[i] = r_2[i];
        }

     
        //condition solved_r by calls to trsv
         for (idx_t i = 0; i < n;i++){
        if (piv_lo[i] != i) {
            auto tmp = solved_r_2[piv_lo[i]];
            solved_r_2[piv_lo[i]] = solved_r_2[i];
            solved_r_2[i] = tmp;
            }
        }


        trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_copy_B, solved_r_2);
        trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_copy_B, solved_r_2);

        //now, using preconditioned r and A, perform GMRES
        //first get ||r|| for GMRES
       
      
        lacpy(tlapack::GENERAL, Zero, Q_B);
        lacpy(tlapack::GENERAL, Zero, H_B);
        normb_B = tlapack::nrm2(solved_r_2);
        //Now initialize first col of Q to be normalized b
        for(int i = 0; i < n; i++) {
            Q_B(i,0) = solved_r_2[i]/normb_B; 
        }
 
        while(num_iter < 15 && inner_res_norm > tol*static_cast<double>(normb_B)) {
            //perform num_iterth step of arnoldi
        
            arnoldi_iter(A_B, H_B, Q_B, LU_copy_B, num_iter);
            num_iter = num_iter + 1; 
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    H_copy_B(i,j) = 0.0;
                }
            }
            tlapack::lacpy(tlapack::GENERAL, H_B, H_copy_B);

            // if(num_iter > 1) return 0.0;
            //solve ||Hx - b||
            
            for(int i = 0; i < n; i++) be_1_2[i] = (i == 0 ? normb_B : static_cast<T2>(0.0));
            if(num_iter != n) Hessenberg_qr(tlapack::slice(H_copy_B,range{0, num_iter+1}, range{0,num_iter}), tlapack::slice(be_1_2,range{0,  num_iter+1}), n);
            else  Hessenberg_qr(H_copy_B, be_1_2, n);
            
            auto da_tmp = tlapack::slice(be_1_2,range{0, num_iter});
            if(num_iter != n) tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS,tlapack::Diag::NonUnit ,tlapack::slice(H_copy_B,range{0, num_iter}, range{0,num_iter}), da_tmp);
            else tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, tlapack::Diag::NonUnit, H_copy_B, da_tmp);
            //our solution vector is now obtained by multiplying by Q_n
            if(num_iter != n) tlapack::gemv(tlapack::NO_TRANS, static_cast<T2>(1.0), tlapack::slice(Q_B,range{0, n}, range{0,num_iter}), tlapack::slice(be_1_2, range{0, num_iter}), static_cast<T2>(0.0), solved_r_2);
            else tlapack::gemv(tlapack::NO_TRANS, static_cast<T2>(1.0), Q_B, be_1_2, static_cast<T2>(0.0), solved_r_2); 
        }
        //update r
        for(int i = 0; i < n; i++) solved_r[i] = static_cast<double>(solved_r_2[i]);
        tlapack::axpy(1.0, solved_r, x);
        for(int i = 0; i < n; i++) x_2[i] = static_cast<T2>(x[i]);

        num_iter = 0;




    } 


    

    while(count < num_total_iter) {
        count = count + 1;
        for(int i = 0; i < n; i++) 
        {
        r[i] = b[i]; 
        r_f[i] = b_f[i];
        }
        tlapack::gemv(NO_TRANS, -1.0, FG, x, 1.0, r);
        for(int i = 0; i < n; i++) r_f[i] = static_cast<float>(r[i]);
        float m;
        res_norm = 0.0;
        for(int i = 0; i < n; i++) {
            m = m > abs(bd[i]) ? m : abs(bd[i]);
            res_norm = res_norm > abs(r[i]) ? res_norm : abs(r[i]);
        }

        res_norm = res_norm/(tlapack::lange(tlapack::INF_NORM, FG)*m);  
        myfile << count << "," << res_norm << std::endl;
        
        //copy r to solved_r
        for(int i = 0; i < n; i++) {
            solved_r_f[i] = r_f[i];
        }
        //condition solved_r by calls to trsv
         for (idx_t i = 0; i < n;i++){
        if (piv_lo[i] != i) {
            auto tmp = solved_r_f[piv_lo[i]];
            solved_r_f[piv_lo[i]] = solved_r_f[i];
            solved_r_f[i] = tmp;
        }
    }

        trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_copy, solved_r_f);
        trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_copy, solved_r_f);

        //now, using preconditioned r and A, perform GMRES
        //first get ||r|| for GMRES
       
      
        lacpy(tlapack::GENERAL, Zero, Q);
        lacpy(tlapack::GENERAL, Zero, H);
        auto normb = tlapack::nrm2(solved_r_f);
        //Now initialize first col of Q to be normalized b
        for(int i = 0; i < n; i++) {
            Q(i,0) = solved_r_f[i]/normb; 
        }
 
        while(num_iter < 10 && inner_res_norm > tol*normb) {
            //perform num_iterth step of arnoldi
        
            arnoldi_iter(A_float, H, Q, LU_copy , num_iter);
            num_iter = num_iter + 1; 
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    H_copy(i,j) = 0.0;
                }
            }
            tlapack::lacpy(tlapack::GENERAL, H, H_copy);

            // if(num_iter > 1) return 0.0;
            //solve ||Hx - b||
            
            for(int i = 0; i < n; i++) be_1_f[i] = (i == 0 ? normb : 0.0);
            if(num_iter != n) Hessenberg_qr(tlapack::slice(H_copy,range{0, num_iter+1}, range{0,num_iter}), tlapack::slice(be_1_f,range{0,  num_iter+1}), n);
            else  Hessenberg_qr(H_copy, be_1_f, n);
            
            auto da_tmp = tlapack::slice(be_1_f,range{0, num_iter});
            if(num_iter != n) tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS,tlapack::Diag::NonUnit ,tlapack::slice(H_copy,range{0, num_iter}, range{0,num_iter}), da_tmp);
            else tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, tlapack::Diag::NonUnit, H_copy, da_tmp);
            //our solution vector is now obtained by multiplying by Q_n
            if(num_iter != n) tlapack::gemv(tlapack::NO_TRANS, 1.0, tlapack::slice(Q,range{0, n}, range{0,num_iter}), tlapack::slice(be_1_f, range{0, num_iter}), 0.0, solved_r_f);
            else tlapack::gemv(tlapack::NO_TRANS, 1.0, Q, be_1_f, 0.0, solved_r_f); 
        }
        //update r
         for(int i = 0; i < n; i++) solved_r[i] = static_cast<double>(solved_r_f[i]);
        tlapack::axpy(1.0, solved_r, x);
        for(int i = 0; i < n; i++) x_f[i] = static_cast<T2>(x[i]);

        num_iter = 0;
        if(count > num_total_iter) break;



    } 


    //declare identity
    std::vector<float> I_(n*n);
    tlapack::LegacyMatrix<float, idx_t> I(n, n, I_.data(), n);
    for(int i = 0; i < n; i++) I(i,i) = 1.0;


    float FR = tlapack::nrm2(x);
    tlapack::axpy(-1.0, bd, x);

    std::cout << "forward err : " << tlapack::nrm2(x)/FR << std::endl;
    std::cout << "backward err ; " << res_norm << std::endl;

    num_iter = 0;
    tlapack::lacpy(tlapack::GENERAL, Zero, Q);
    tlapack::lacpy(tlapack::GENERAL, Zero, H);
    auto normb = tlapack::nrm2(b);
    for(int i = 0; i < n; i++) {
        Q(i,0) = b[i]/normb;
    }
    while(num_iter < 50 && inner_res_norm > tol*normb) {
            //perform num_iterth step of arnoldi
            
            arnoldi_iter(A_float, H, Q, I ,num_iter);
            num_iter = num_iter + 1; 
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    H_copy(i,j) = 0.0;
                }
            }
            tlapack::lacpy(tlapack::GENERAL, H, H_copy);

            // if(num_iter > 1) return 0.0;
            //solve ||Hx - b||
            
            for(int i = 0; i < n; i++) be_1[i] = (i == 0 ? normb : 0.0);
            if(num_iter != n) Hessenberg_qr(tlapack::slice(H_copy,range{0, num_iter+1}, range{0,num_iter}), tlapack::slice(be_1,range{0,  num_iter+1}), n);
            else  Hessenberg_qr(H_copy, be_1, n);
            
            auto da_tmp = tlapack::slice(be_1,range{0, num_iter});
            if(num_iter != n) tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS,tlapack::Diag::NonUnit ,tlapack::slice(H_copy,range{0, num_iter}, range{0,num_iter}), da_tmp);
            else tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, tlapack::Diag::NonUnit, H_copy, da_tmp);
            //our solution vector is now obtained by multiplying by Q_n
            if(num_iter != n) tlapack::gemv(tlapack::NO_TRANS, 1.0, tlapack::slice(Q,range{0, n}, range{0,num_iter}), tlapack::slice(be_1, range{0, num_iter}), 0.0, solved_r);
            else tlapack::gemv(tlapack::NO_TRANS, 1.0, Q, be_1, 0.0, solved_r); 
        }

        //find residual -
        tlapack::gemv(tlapack::NO_TRANS, 1.0, FG, solved_r, -1.0, b);
        double dm;
        double dres_norm = 0.0;
        for(int i = 0; i < n; i++) {
            dm = dm > abs(bd[i]) ? dm : abs(bd[i]);
            dres_norm = dres_norm > abs(b[i]) ? dres_norm : abs(b[i]);
        }
        std::cout << "GMRES without precond gives err : " << tlapack::nrm2(b)/(tlapack::lange(INF_NORM, FG)*dm) << std::endl;


    return res_norm;




}




int main(int argc, char** argv) {
     typedef ml_dtypes::float8_e4m3fn float8e4m3fn;
    typedef ml_dtypes::float8_e5m2 float8e5m2;
    typedef ml_dtypes::float8_ieee_p<4> floate4m3;
    typedef ml_dtypes::float8_ieee_p<3> floate5m2;
    typedef ml_dtypes::block_float8_ieee<4> bfp;
    typedef Eigen::bfloat16 bfloat16;
    typedef Eigen::half half;
    int m, n;

    std::cout << std::scientific << std::endl;
    
    // Default arguments
    n = atoi(argv[1]);
    float err1 = 0;
    float er3 = 0;
    float err2 = 0;
    int num_iter_1 = atoi(argv[6]);
    int total_num_iter = 5;
    if(argc > 7) total_num_iter = atoi(argv[7]);
   
    float a = 0.0;
    float b = 0.0;
    float c = 0.0;

   
    er3 += run(n, 1.0, static_cast<float>(atoi(argv[3])), atoi(argv[8]), atoi(argv[4]), num_iter_1, total_num_iter, a, b, c);    
        bool verified;
    if (er3 <= std::pow(10,-6)) {
      verified = true;
      std::cout << " VERIFICATION SUCCESSFUL\n";
      std::cout << " Error is " <<  er3 << std::endl;
  } else {
      verified = false;
      std::cout << " VERIFICATION FAILED\n";
      std::cout << " Error is " <<  er3 << std::endl;
  }
  FILE *fp = fopen("./log.txt", "w");
  fputs(verified ? "true\n" : "false\n", fp);
  fprintf(fp, "%20.13E\n", static_cast<double>(er3));

  // record time
  FILE *fp_t = fopen("./time.txt", "w");
  fprintf(fp_t, "%f\n", 1);
   
   
    return 0;
}