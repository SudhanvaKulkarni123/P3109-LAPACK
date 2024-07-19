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
#include "getrf_blocked.hpp"
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

using namespace Eigen;
using namespace ml_dtypes;

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

//-------------------------------------------------------------------

template<typename matrix_t>
void isZero(const matrix_t& A)
{
    using real_t = type_t<matrix_t>;
    for(int i = 0; i < nrows(A); i++){
        for(int j = 0; j < ncols(A); j++){
            if(A(i,j) != real_t(0)) { std::cout << "Not zero mat" << std::endl; return;}
        }
    }
    std::cout << "Zero mat" << std::endl;

}

template<typename matrix_t>
void PrintAllExp(matrix_t& A, Layout L)
{   
    using real_t = type_t<matrix_t>;
    
    real_t maxim = real_t(0);
    if(L == Layout::ColMajor){
    std::cout << "Printing all column exponents" << std::endl;
    for(int j = 0; j < ncols(A); j++) {
    for(int i = 0; i < nrows(A); i++) {
            if (abs(A(i,j)) > maxim) maxim = abs(A(i,j));
        }
        std::cout << int(floor((log2(maxim)))) << std::endl;
        maxim = real_t(0);
    
    }
    }
    else if(L == Layout::RowMajor){
    std::cout << "Printing all row exponents" << std::endl;
    for(int i = 0; i < ncols(A); i++) {
        for(int j = 0; j < nrows(A); j++) {
            if (abs(A(j,i)) > maxim) maxim = abs(A(j,i));
        }
        std::cout << floor(float(log2(maxim))) << std::endl;
        real_t maxim = real_t(0);
    }
    }
    std::cout << "End of exponents" << std::endl;

    return;

}

template<typename T>
T get_machine_epsilon()
{
    double eps = 1.0;
    while (static_cast<T>(1.0 + 0.5 * eps) != static_cast<T>(1.0))
        eps *= (0.5);
    return static_cast<T>(eps);
}

template <typename matrix_t>
bool check_matrix(const matrix_t& A)
{
    using T = type_t<matrix_t>;
    for(int i = 0; i < nrows(A); i++){
        for(int j = 0; j < ncols(A); j++){
            if(abs(A(i,j)) > T(1.0)) return false;
        }
    }

    return true;
}









template<typename vectr_t>
void printVector(vectr_t& v)
{
    using idx_t = tlapack::size_type<vectr_t>;
    const idx_t n = tlapack::size(v);

    for (idx_t i = 0; i < n; ++i)
        std::cout << v[i] << " ";
    std::cout << std::endl;
}




//------------------------------------------------------------------------------

#ifdef PY_SSIZE_T_CLEAN
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

#else
 std::vector<float> getFileNames(const std::string& directory, float& true_cond) {
        std::vector<float> fileNames;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string fileName = entry.path().filename().string();
                fileName = fileName.substr(7, fileName.size() - 11); 
                true_cond = std::stof(fileName);
                fileNames.push_back(true_cond);
            }
        }
        return fileNames;
    }

float find_closest(const std::vector<float>& fileNames, float true_cond) {
    float min_diff = std::abs(fileNames[0] - true_cond);
    float closest = fileNames[0];
    for (int i = 1; i < fileNames.size(); i++) {
        float diff = std::abs(fileNames[i] - true_cond);
        if (diff < min_diff) {
            min_diff = diff;
            closest = fileNames[i];
        }
    }
    return closest;
}

template<typename T, typename matrix_t>
int constructMatrix(int n, float cond, int space, bool geom, matrix_t& A, int p, float& true_cond) {
   

    std::vector<float> fileNames = getFileNames("/root/home/Precimonious/GMRES_IR/tempscripts/matrix_collection_" + std::to_string(p), true_cond);
    true_cond = find_closest(fileNames, cond);
    std::string fileName = "/root/home/Precimonious/GMRES_IR/tempscripts/matrix_collection_" + std::to_string(p) + "/" + "matrix_" + std::to_string(static_cast<int>(true_cond)) + ".csv";
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "File not found" << std::endl;
        return 1;
    }
    std::string line;
    int i = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        int j = 0;
        while (std::getline(ss, token, ',')) {
            if(j == n && i == n) break;
            else {
            A(i, j) = std::stod(token);
            j++;
            if(j == n) {i++; j = 0;}
            }
        }
    }
    std::cout << "true_cond is : " <<  true_cond << std::endl;
    return 0;

}

#endif

template<typename T>
T inline closest_power_of_two(T n) {
    return pow(2, floor(log2(n)));
}


std::ofstream myfile("e5m2_error_f_cond.csv");
std::ofstream timefile("time.txt");
//each T_2 is the precision in which first iter of GMRES-IR will take place, T3 is the initial solution precision
template <typename T2, typename T3, typename T4, typename T5, typename T6>
float GMRES_IR(size_t n, ml_dtypes::float8_ieee_p<4> scale, float cond, int p,  int variant = 0, int num_iter_1 = 0, int num_total_iter = 5, kernel_type precond_mode = kernel_type::RIGHT_LU, int max_gmres_iter = 20, int inner_num_gmres_iter = 20, int num_IR_iter=20, refinement_type method = refinement_type::GMRES_IR) 
{   
    using matrix_t = tlapack::LegacyMatrix<float>;
    using block_mat_t = tlapack::BlockMatrix<float>;
    using real_t = tlapack::real_type<float>;
    using idx_t = size_t;
    using range = pair<idx_t, idx_t>;

    double d_conv_bound = sqrt(static_cast<double>(n)) * std::pow(2, -53);
    double s_conv_bound = sqrt(static_cast<double>(n)) * std::pow(2, -24);   

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Create the n-by-n matrix A in precision ml_dtypes::float8_ieee_p<4>
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
    constructMatrix<float>(n, cond, std::ceil(cond/static_cast<float>(5)) > n-1 ? n-1 : std::ceil(cond/static_cast<float>(5)) , true, FG, p, true_cond);
   
 
    std::cout << "Matrix has entries between +1 and -1 : " << check_matrix(FG) << std::endl;
    // std::vector<float> S_scal(n*n, 0.0);
    // for (size_t i = 0; i < n; i++){
    //     S_scal[i] = 1.0;
    // }
    // std::vector<float> R_scal(n*n, 0.0);
    // for (size_t i = 0; i < n; i++){
    //     R_scal[i] = 1.0;
    // }

    // float maxR, maxS;
   
    //     for(int i = 0; i < n; i++){
    //         auto c1 = tlapack::rows(FG,range(i,i+1));
    //         R_scal[i] = 1/closest_power_of_two((tlapack::lange(tlapack::Norm::Inf, c1)));
    //          R_scal[i] = 1.0;

    //     }
    //     for(int j = 0; j < n; j++){
    //         for(int k = 0; k < n; k++){
    //             FG_d(j,k) = FG(j,k)*(R_scal[j]);
    //         }
    //     }
    //       for(int i = 0; i < n; i++){
    //         auto c1 = tlapack::cols(FG,range(i,i+1));
    //         S_scal[i] = 1/closest_power_of_two((tlapack::lange(tlapack::Norm::Inf, c1)));
    //          S_scal[i] = 1.0;
    //     }
    //     for(int j = 0; j < n; j++){
    //         for(int k = 0; k < n; k++){
    //             FG_d(k,j) = FG(k,j)*(S_scal[j]);
    //         }
    //     }
        

    //     //next we need to scale by a parameter theta
        double maxA = tlapack::lange(tlapack::Norm::Max, FG);

        
        double normA = tlapack::lange(tlapack::Norm::Inf, FG);

    //first we'll get cond aafter preconditioning


    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            FG(i,j) = FG(i,j);
        }
    }

    tlapack::lacpy(tlapack::Uplo::General, FG, A);


   


    for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < n; ++i){
            //A(i,j) = static_cast<real_t>(sqrt(float(scale)*0.125)*FG(i,j)/normA);
            A_F(i,j) = static_cast<T6>(FG(i,j));
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
    std::vector<float> x_1_(n);
    tlapack::LegacyVector<float, idx_t> x_1(n, x_1_.data());
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

    std::vector<float> b_1_(n);
    tlapack::LegacyVector<float, idx_t> b_1(n, b_1_.data());

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

    std::vector<float> r_1_(n);
    tlapack::LegacyVector<float, idx_t> r_1(n, r_1_.data());

    std::vector<T2> r_2_(n);
    tlapack::LegacyVector<T2, idx_t> r_2(n, r_2_.data());

    std::vector<T3> r_3_(n);
    tlapack::LegacyVector<T3, idx_t> r_3(n, r_3_.data());

    std::vector<T6> r_f_(n);
    tlapack::LegacyVector<T6, idx_t> r_f(n, r_f_.data());

    std::vector<double> solved_r_(n);
    tlapack::LegacyVector<double, idx_t> solved_r(n, solved_r_.data());

    std::vector<float> solved_r_1_(n);
    tlapack::LegacyVector<float, idx_t> solved_r_1(n, solved_r_1_.data());

    std::vector<T2> solved_r_2_(n);
    tlapack::LegacyVector<T2, idx_t> solved_r_2(n, solved_r_2_.data());

    std::vector<T3> solved_r_3_(n);
    tlapack::LegacyVector<T3, idx_t> solved_r_3(n, solved_r_3_.data());

    std::vector<T6> solved_r_f_(n);
    tlapack::LegacyVector<T6, idx_t> solved_r_f(n, solved_r_f_.data());



    std::vector<double> be_1_(n);
    tlapack::LegacyVector<double, idx_t> be_1(n, be_1_.data());

    std::vector<float> be_1_1_(n);
    tlapack::LegacyVector<float, idx_t> be_1_1(n, be_1_1_.data());

    std::vector<T2> be_1_2_(n);
    tlapack::LegacyVector<T2, idx_t> be_1_2(n, be_1_2_.data());

    std::vector<T3> be_1_3_(n);
    tlapack::LegacyVector<T3, idx_t> be_1_3(n, be_1_3_.data());

    
    std::vector<T6> be_1_f_(n);
    tlapack::LegacyVector<T6, idx_t> be_1_f(n, be_1_f_.data());


    for( int i = 0; i < n; i++) {
        b[i] = static_cast<double>((-0.5*(static_cast<double>(rand()))*double(scale)/static_cast<double>(RAND_MAX)));
        b[i] += static_cast<double>((static_cast<double>(rand())*double(scale)/static_cast<double>(RAND_MAX)));
        b_f[i] = static_cast<float>(b[i]);
        b_1[i] = static_cast<float>(b[i]);
        b_2[i] = static_cast<T2>(b[i]);
        b_3[i] = static_cast<T3>(b[i]);
        be_1[i] = (i == 0 ? 1.0 : 0.0);
        be_1_f[i] = static_cast<T6>(be_1[i]);
        be_1_1[i] = static_cast<float>(be_1[i]);
        be_1_2[i] = static_cast<T2>(be_1[i]);
        be_1_3[i] = static_cast<T3>(be_1[i]);
        bd[i] = static_cast<double>(b[i]);
    }

    //perform LU on A and FG
    std::vector<float> LU_(n * n);
    std::vector<int> LU_exp(n, 0);
    //tlapack::BlockMatrix<float> LU(n, n, LU_.data(), n, LU_exp.data(), n);
    tlapack::LegacyMatrix<float, idx_t> LU(n, n, LU_.data(), n);

    //keep copies in float and the second precision
    std::vector<T6> LU_copy_(n * n);
    tlapack::LegacyMatrix<T6, idx_t> LU_copy(n, n, LU_copy_.data(), n);

    std::vector<T2> LU_copy_B_(n * n);
    tlapack::LegacyMatrix<T2, idx_t> LU_copy_B(n, n, LU_copy_B_.data(), n);

    std::vector<T3> LU_copy_C_(n * n);
    tlapack::LegacyMatrix<T3, idx_t> LU_copy_C(n, n, LU_copy_C_.data(), n);

    std::vector<T4> LU_copy_D_(n * n);
    tlapack::LegacyMatrix<T4, idx_t> LU_copy_D(n, n, LU_copy_D_.data(), n);

    std::vector<T5> LU_copy_E_(n * n);
    tlapack::LegacyMatrix<T5, idx_t> LU_copy_E(n, n, LU_copy_E_.data(), n);


    std::vector<double> LU_double_(n * n);
    tlapack::LegacyMatrix<double, idx_t> LU_double(n, n, LU_double_.data(), n);

    //we need H is float and T2
    std::vector<T6> H_(n*n);
    tlapack::LegacyMatrix<T6, idx_t> H(n, n, H_.data(), n);

    std::vector<T2> H_B_(n*n);
    tlapack::LegacyMatrix<T2, idx_t> H_B(n, n, H_B_.data(), n);


    std::vector<T6> H_copy_(n*n);
    tlapack::LegacyMatrix<T6, idx_t> H_copy(n, n, H_copy_.data(), n);

    std::vector<T2> H_copy_B_(n*n);
    tlapack::LegacyMatrix<T2, idx_t> H_copy_B(n, n, H_copy_B_.data(), n);

    std::vector<T6> Q_(n*n);
    tlapack::LegacyMatrix<T6, idx_t> Q(n, n, Q_.data(), n);

    std::vector<T2> Q_B_(n*n);
    tlapack::LegacyMatrix<T2, idx_t> Q_B(n, n, Q_B_.data(), n);

    std::vector<T6> Z_(n*n);
    tlapack::LegacyMatrix<T6, idx_t> Z(n, n, Z_.data(), n);

    std::vector<T2> Z_B_(n*n);
    tlapack::LegacyMatrix<T2, idx_t> Z_B(n, n, Z_B_.data(), n);

    std::vector<T6> cs_(n);
    tlapack::LegacyVector<T6, idx_t> cs(n, cs_.data());

    std::vector<T2> cs_B_(n);
    tlapack::LegacyVector<T2, idx_t> cs_B(n, cs_B_.data());

    std::vector<T6> sn_(n);
    tlapack::LegacyVector<T6, idx_t> sn(n, sn_.data());

    std::vector<T2> sn_B_(n);
    tlapack::LegacyVector<T2, idx_t> sn_B(n, sn_B_.data());

    tlapack::lacpy(tlapack::GENERAL, A, LU);
    



    tlapack::lacpy(tlapack::GENERAL, FG, LU_double);

    std::cout << "successfully copied matrices" << std::endl;
   
    //declare arrays for piv
    std::vector<size_t> piv_lo(n);
    std::vector<size_t> piv_hi(n);


    int info, infotoo;

    
  
    // if(variant == 0) info = tlapack::getrf(LU, piv_lo, tlapack::GetrfOpts{GetrfVariant::Recursive});
    // else info = tlapack::getrf(LU, piv_lo, tlapack::GetrfOpts{GetrfVariant::Level0});
    info = getrf_blocked(LU, piv_lo);
    std::cout << "LU factorization done" << std::endl;

    if (info != 0) {
        std::cerr << "Matrix could not be factorized :(" << std::endl;
        return -1;
    }
    
    int infotoo2 = tlapack::getrf(LU_double, piv_hi, tlapack::GetrfOpts{GetrfVariant::Level0});
    if (infotoo2 != 0) {
        std::cerr << "Matrix could not be factorized in fp64 :(" << std::endl;
        return -1;
    }

    std::cout << " factorization in double complete" << std::endl;
    if(n==8) {printMatrix(LU_double); std::cout << "----------" << std::endl; printMatrix(LU);}
    
    // #undef USE_LEGACY

    // PrintAllExp(LU_double, Layout::ColMajor);
    // PrintAllExp(LU_double, Layout::RowMajor);

    // //compute sol in double

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

   

  
    std::cout << std::endl;
    tlapack::lacpy(tlapack::Uplo::General, LU, LU_copy);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            LU_copy_B(i,j) = static_cast<T2>(LU_copy(i,j));
            LU_copy_C(i,j) = static_cast<T3>(LU_copy(i,j));
            LU_copy_D(i,j) = static_cast<T4>(LU_copy(i,j));
            LU_copy_E(i,j) = static_cast<T5>(LU_copy(i,j));
            
        }
    }



    double total_time = 0.0;

    //for(int i = 0; i < n; i++) b_3[i] = b_3[i]*static_cast<T3>(R_scal[i]); 



    if(method == refinement_type::GMRES_IR) 
    {

    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_copy_C, b_3);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_copy_C, b_3);
    //init soln is now in b_3    

    //for(int i = 0; i < n; i++) b_3[i] = b_3[i]*static_cast<T3>(S_scal[i]);



    
    //now we can begin the actual IR
    double res_norm = 1.0;
    double inner_res_norm = 1.0;

    int count = 0; //to keep track of number of IR iterations
    T2 normb_B = static_cast<T2>(0.0);

    int num_iter = 0; //number of iters for GMRES
    float tol = std::pow(10,-6);
    for(int i = 0; i < n; i++) x[i] = static_cast<double>(b_3[i]);
    for(int i = 0; i < n; i++) x_2[i] = static_cast<T2>(b_3[i]);

    //this is the first iteration of IR that uses a lower precision

    std::vector<T2> xx_2_(n);
    std::vector<T6> xx_f_(n);
    tlapack::LegacyVector<T2, idx_t> xx_2(n, xx_2_.data());
    tlapack::LegacyVector<T6, idx_t> xx_f(n, xx_f_.data());
    for(int i = 0; i < n; i++) {
        xx_2[i] = static_cast<T2>(0.0);
        xx_f[i] = static_cast<T6>(0.0);
    }
    while(count < num_iter_1)
    {
    compute_residual(FG, x, b, r);
    std::cout << " L2 norm of residual is : " << tlapack::nrm2(r) << std::endl;
    std::cout << "L-inf norm of residual is : " << inf_norm(r) << std::endl;
    double norm_x = inf_norm(x);
    myfile << count << "," << inf_norm(r)/(normA*norm_x) << std::endl;
        if(inf_norm(r)/(normA*norm_x) < d_conv_bound) {
            std::cout << "converged to double precision" << std::endl;
            break;
        }
        else if(inf_norm(r)/(normA*norm_x) < s_conv_bound) {
            std::cout << "converged to single precision" << std::endl;
        }
    // if(inf_norm(r)/(normA*inf_norm(x)) < s_conv_bound) {
    //     std::cout << "converged to single precision" << std::endl;
    //     break;
    // }
    

    for(int i =0; i < n; i++) 
    {
        r_2[i] = static_cast<T2>(r[i]);
        solved_r_2[i] = static_cast<T2>(r[i]);
        if(precond_mode == kernel_type::RIGHT_GMRES) {
            r_f[i] = static_cast<T6>(r[i]);
            solved_r_f[i] = static_cast<T6>(r[i]);
        }
    }
    
    if(precond_mode == kernel_type::RIGHT_GMRES) {
        for(int i = 0; i < n; i++) solved_r_f[i] = static_cast<T6>(solved_r_2[i]);
    }

    if(precond_mode == kernel_type::RIGHT_LU) {
        GMRES(A_B, Q_B, H_B, LU_copy_B, piv_lo, r_2, solved_r_2, xx_2, cs_B, sn_B, kernel_type::RIGHT_LU, max_gmres_iter, num_iter_1);
    }
    else if(precond_mode == kernel_type::LEFT_LU) {
        GMRES(A_B, Q_B, H_B, LU_copy_B, piv_lo, r_2, solved_r_2, xx_2, cs_B, sn_B, kernel_type::LEFT_LU, max_gmres_iter, num_iter_1);
        std::cout << "kill me" << std::endl;
    }
    else if(precond_mode == kernel_type::RIGHT_GMRES) {
        FGMRES(A_F, A_B, Q, Q_B, H , H_B, Z, LU_copy, LU_copy_B, piv_lo, r_f, solved_r_f, xx_f, solved_r_f, cs, sn, cs_B, sn_B, kernel_type::RIGHT_GMRES, max_gmres_iter, num_iter_1, 1e-8, inner_num_gmres_iter, num_IR_iter);
    }
    if(precond_mode == kernel_type::RIGHT_GMRES) {
        for(int i = 0; i < n; i++) x[i] = x[i] + static_cast<double>(xx_f[i]);
    }
    else {
        for(int i = 0; i < n; i++) x[i] = x[i] + static_cast<double>(xx_2[i]);
        num_lower_prec_gmres_loops += num_higher_prec_gmres_loops;
        num_higher_prec_gmres_loops = 0;
    }
    count++;

    myfile << count << "," << inf_norm(r) << std::endl;
    }

    //if mode is right_gmres, we are done. else we need to finish the higherprecision IR
    if(precond_mode != kernel_type::RIGHT_GMRES)
    {
        while(count < num_total_iter)
        {
        compute_residual(FG, x, b, r);
        std::cout << " L2 norm of residual is : " << tlapack::nrm2(r) << std::endl;
        std::cout << "L-inf norm of residual is : " << inf_norm(r) << std::endl;
        double norm_x = inf_norm(x);
        myfile << count << "," << inf_norm(r)/(normA*norm_x) << std::endl;
        if(inf_norm(r)/(normA*norm_x) < d_conv_bound) {
            std::cout << "converged to double precision" << std::endl;
            break;
        }
        else if(inf_norm(r)/(normA*norm_x) < s_conv_bound) {
            std::cout << "converged to single precision" << std::endl;
        }
        for(int i =0; i < n; i++) 
        {
            r_f[i] = static_cast<T6>(r[i]);
            solved_r_f[i] = static_cast<T6>(r[i]);
        }
        LU_solve(LU_copy, piv_lo, solved_r_f);

        if(precond_mode == kernel_type::RIGHT_LU) {
            GMRES(A_F, Q, H, LU_copy, piv_lo, r_f, solved_r_f, xx_f, cs, sn, kernel_type::RIGHT_LU, max_gmres_iter, num_iter_1);       
        }
        else if(precond_mode == kernel_type::LEFT_LU) {
            GMRES(A_F, Q, H, LU_copy, piv_lo, r_f, solved_r_f, xx_f, cs, sn, kernel_type::LEFT_LU, max_gmres_iter, num_iter_1);
        }
        for(int i = 0; i < n; i++) x[i] = x[i] + static_cast<double>(xx_f[i]);
        count++;
        
        }
    }


    double norm_x = inf_norm(x);

    tlapack::gemv(tlapack::NO_TRANS, 1.0, FG, x, -1.0, b);
        res_norm = inf_norm(b)/(normA*norm_x);
        if (res_norm < d_conv_bound) std::cout << "converged to  double precision" << std::endl;
        else if(res_norm < s_conv_bound) std::cout << "converged to single precision" << std::endl;
        else std::cout << "did not converge" << std::endl;

    
    for(int i = 0; i < n; i++) x[i] = x[i] - bd[i];

    std::cout << "forward error is : " << inf_norm(x)/inf_norm(bd) << std::endl;

    
    auto normb = tlapack::nrm2(b);

        //find residual -
        
        

    timefile << total_time << std::endl;

    if(res_norm == 0.0) res_norm = 999999.0;
    std::cout << "backward err ; " << res_norm << std::endl;
    std::cout << "total number of higher prec gmres loops : " << num_higher_prec_gmres_loops << std::endl;
    std::cout << "total number of lower prec gmres loops : " << num_lower_prec_gmres_loops << std::endl;
    return res_norm;
    }
    else if(method == refinement_type::NVIDIA_IR)
    {
    //the "NVidia method" just directly runs preconditioned GMRES to solve Ax = b rather than solving Ax = r and dealing with all the updating yapping. Only bad part is that it needs to be in double
    int count = 0;

    for(int i = 0; i < n; i++)
    {
        x[i] = static_cast<double>(b[i]);
        r_3[i] = static_cast<T2>(b[i]);
        r_f[i] = static_cast<T6>(b[i]);
    }
    if(precond_mode == kernel_type::LEFT_LU)    LU_solve(LU_copy_C, piv_lo, r_3);

    for(int i = 0; i < n; i++) r_2[i] = static_cast<T2>(r_3[i]);

    std::vector<T2> xx_2_(n);
    std::vector<T6> xx_f_(n);
    tlapack::LegacyVector<T2, idx_t> xx_2(n, xx_2_.data());
    tlapack::LegacyVector<T6, idx_t> xx_f(n, xx_f_.data());
    for(int i = 0; i < n; i++) {
        xx_2[i] = static_cast<T2>(0.0);
        xx_f[i] = static_cast<T6>(0.0);
    }


    while(count < num_iter_1)
    {

    for(int i =0; i < n; i++) 
    {
        if(precond_mode == kernel_type::RIGHT_GMRES) {
            r_f[i] = static_cast<T6>(b[i]);
            solved_r_f[i] = static_cast<T6>(b[i]);
        }
    }
    
    if(precond_mode == kernel_type::RIGHT_GMRES) {
        for(int i = 0; i < n; i++) solved_r_f[i] = static_cast<T6>(solved_r_2[i]);
    }

    if(precond_mode == kernel_type::RIGHT_LU) {
        GMRES(A_B, Q_B, H_B, LU_copy_B, piv_lo, r_2, solved_r_2, xx_2, cs_B, sn_B, kernel_type::RIGHT_LU, max_gmres_iter, num_iter_1);
    }
    else if(precond_mode == kernel_type::LEFT_LU) {
        GMRES(A_B, Q_B, H_B, LU_copy_B, piv_lo, r_2, solved_r_2, xx_2, cs_B, sn_B, kernel_type::LEFT_LU, max_gmres_iter, num_iter_1);
    }
    else if(precond_mode == kernel_type::RIGHT_GMRES) {
        FGMRES(A_F, A_B, Q, Q_B, H , H_B, Z, LU_copy, LU_copy_B, piv_lo, r_f, solved_r_f, xx_f, solved_r_f, cs, sn, cs_B, sn_B, kernel_type::RIGHT_GMRES, max_gmres_iter, num_iter_1, 1e-8, inner_num_gmres_iter, num_IR_iter);
    }
    if(precond_mode == kernel_type::RIGHT_GMRES) {
        for(int i = 0; i < n; i++) x[i] = static_cast<double>(xx_f[i]);
    }
    else {
        for(int i = 0; i < n; i++) x[i] = static_cast<double>(xx_2[i]);
    }
    count++;
    }
    double norm_x = inf_norm(x);
    double res_norm = 0.0;
    tlapack::gemv(tlapack::NO_TRANS, 1.0, FG, x, -1.0, b);
        res_norm = inf_norm(b)/(normA*norm_x);
        if (res_norm < d_conv_bound) std::cout << "converged to  double precision" << std::endl;
        else if(res_norm < s_conv_bound) std::cout << "converged to single precision" << std::endl;
        else std::cout << "did not converge" << std::endl;

    
    for(int i = 0; i < n; i++) x[i] = x[i] - bd[i];

    std::cout << "forward error is : " << inf_norm(x)/inf_norm(bd) << std::endl;

    
    auto normb = tlapack::nrm2(b);

        //find residual -
        
        

    timefile << total_time << std::endl;

    if(res_norm == 0.0) res_norm = 999999.0;
    std::cout << "backward err ; " << res_norm << std::endl;
    std::cout << "total number of higher prec gmres loops : " << num_higher_prec_gmres_loops << std::endl;
    std::cout << "total number of lower prec gmres loops : " << num_lower_prec_gmres_loops << std::endl;
    return res_norm;

    } 
    else {
        //just find soln with no IR

        for(int i = 0; i < n; i++) b_3[i] = static_cast<T3>(b[i]);

         for (idx_t i = 0; i < n;i++){
        if (piv_lo[i] != i) {
            auto tmp = b_3[piv_lo[i]];
            b_3[piv_lo[i]] = b_3[i];
            b_3[i] = tmp;               
        }
    }


       

    tlapack::trsv(Uplo::Lower, tlapack::NO_TRANS, Diag::Unit, LU_copy_C, b_3);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++) {
            LU_copy_C(i,j) = LU_copy_C(i,j);
        }
    }
    tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, Diag::NonUnit, LU_copy_C, b_3);

    for(int i = 0; i < n; i++) x[i] = static_cast<double>(b_3[i]);

    double res_norm;
    double norm_x = inf_norm(x);
    tlapack::gemv(tlapack::NO_TRANS, 1.0, FG, x, -1.0, b);
        res_norm = inf_norm(b)/(normA*norm_x);
    std::cout << "backward err ; " << res_norm << std::endl;

    for(int i = 0; i < n; i++) x[i] = x[i] - bd[i];
    std::cout << "forward error is : " << inf_norm(x)/inf_norm(bd) << std::endl;

    }

    return 0.0;



}



int test_Block_matrix_mult()
{
   int n  = 20;
   using T = ml_dtypes::float8_ieee_p<4>;
   std::vector<ml_dtypes::float8_ieee_p<4>> LU_(n * n);
    std::vector<int> LU_exp(n, 0);
    for(int i = 0; i < n; i++) LU_exp[i] = 1;
    tlapack::BlockMatrix<ml_dtypes::float8_ieee_p<4>> LU(n, n, LU_.data(), n, LU_exp.data(), n);
    std::vector<ml_dtypes::float8_ieee_p<4>> PU_(n * n);
    std::vector<int> PU_exp(n, 0);
    tlapack::BlockMatrix<ml_dtypes::float8_ieee_p<4>> PU(n, n, PU_.data(), n, PU_exp.data(), n);
    std::vector<ml_dtypes::float8_ieee_p<4>> KU_(n * n);
    std::vector<int> KU_exp(n, 0);
    tlapack::BlockMatrix<ml_dtypes::float8_ieee_p<4>> KU(n, n, KU_.data(), n, KU_exp.data(), n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            LU(i,j) = 1.0;
            PU(i,j) = 1.0;
            KU(i,j) = 1.0;
        }
    }

    std::vector<float> FG_(n * n);
    tlapack::LegacyMatrix<float, int> FG(n, n, FG_.data(), n);

    gemm(tlapack::NO_TRANS, tlapack::NO_TRANS, T(1.0), LU, PU, T(1.0), KU);
    lacpy(tlapack::Uplo::General, KU, FG);

    printMatrix(FG);

    return 0;



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
    float er3 = 0;
    int max_gmres_iter = atoi(argv[2]);
    int num_iter_1 = atoi(argv[6]);
    int total_num_iter = 5;
    if(argc > 7) total_num_iter = atoi(argv[7]);
    kernel_type final;
    refinement_type method = refinement_type::GMRES_IR;
    auto arg_as_str = string(argv[9]);
    if(arg_as_str == string("RIGHT_LU")) final = kernel_type::RIGHT_LU;
    if(arg_as_str == string("LEFT_LU")) final = kernel_type::LEFT_LU;
    if(arg_as_str == string("RIGHT_GMRES")) final = kernel_type::RIGHT_GMRES;
    int inner_gmres_num = atoi(argv[5]);
    int num_IR_iter = atoi(argv[10]);
    if(argc >= 10)
    {
    auto mode_arg = string(argv[11]);
    if(mode_arg == string("NVIDIA")) method = refinement_type::NVIDIA_IR;
    else if(mode_arg == string("GMRES")) method = refinement_type::GMRES_IR;
    else if(mode_arg == string("NO_IR")) method = refinement_type::NO_IR;

    blocks = atoi(argv[12]);
    }
    if(blocks) std::cout << " using BFP " << std::endl;
 
    //test_Block_matrix_mult();
    std::cout << " n : " << n << " maximum number of iterations for GMRES in GMRES-IR : " << max_gmres_iter << "maximum number of iteration for GMRES-IR in first precision : " << num_iter_1 << "maximum number of iterations in second precision : " << total_num_iter - num_iter_1 << " max number of iterations for : " << inner_gmres_num << std::endl;
    er3 += GMRES_IR<float, float, double, double, double>(n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<4>::max()/floate4m3{4.0}, static_cast<float>(atoi(argv[3])), atoi(argv[8]), atoi(argv[4]), num_iter_1, total_num_iter, final, max_gmres_iter, inner_gmres_num, num_IR_iter, method);    
    
    
    bool verified = abs(er3) < 1e-8;
    FILE *fp = fopen("./log.txt", "w");
    fputs(verified ? "true\n" : "false\n", fp);
    fprintf(fp, "%20.13E\n", static_cast<double>(er3));
    std::cout << "err3 : " << er3 << std::endl;
    return 0;
}