//test file for rbf_chol
#include <set>
#include <queue>
#include <algorithm>
#include "tlapack/base/utils.hpp"
#include "tlapack/plugins/legacyArray.hpp"
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
#include "RBF_chol.hpp"
#include "lo_float.h"
#include "lo_float_sci.hpp"

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
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/blas/gemv.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/blas/trsm.hpp>
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
#include "python_utils.hpp"
#include "json_utils.hpp"
#include "num_ops.hpp"
#include "pivoted_cholesky.hpp"
#include "CG_IR.hpp"

#include "num_ops.hpp"
#include "pivoted_cholesky.hpp"
#include "CG_IR.hpp"

using namespace tlapack;

template<typename Vector>
float dot(Vector& a, Vector& b) {
    float sum = 0.0;
    for(int i = 0; i < size(a); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

//function to generate column i of N-by-N A where A is RBF kernel-

//function to compute A(S, :)A(:, S) + lambda*N*A(S,S) where S is some set of indices and A is N*N RBF kernel (elems genrated on the fly)
//and N is the number of training points
//and lambda is a regularization parameter
/// @param A : workspace to store final result of computation
/// @param S, s : set of indices to compute A(S, :)A(:, S) + lambda*N*A(S,S) and  |S|
/// @param N : number of training points
/// @param lambda : regularization parameter
template<typename Matrix, typename Set, typename Data>
void RBF_compressed(Matrix& A, Set& S, int s, int N, float lambda, int img_size, float sigma, DAta& points)
{
    for(int i = 0; i < s; i++) {
        for(int j = 0; j < s; j++) {

            A(i,j) = 0.0f;
            for(auto elem : S) {
                A(i,j) += rbf_value(points, i, elem, sigma, img_size)*rbf_value(points, elem, j, sigma, img_size);
            }
            A(i,j) += lambda*N*rbf_value(points, i, j, sigma, img_size);
        }
    }
    
}


template<typename Data, typename Vector>
void transform_label(Data& points, Vector& y, int N, float sigma, int s)
{
    for(int i = 0; i < N; i++) {
        y[i] = 0.0;
        for(int j = 0; j < s; j++) {
            y[i] += rbf_value(points, i, j, sigma, img_size)*y[i];
            y_cnt++;
        }

    }
}


int main(int argc, char** argv) {

    if(argc < 5) {
        std::cout << "Usage: " << argv[0] << " <n> <k> <r> <sigma>" << std::endl;
        return 1;
    }


    int n = std::stoi(argv[1]);
    int k = std::stoi(argv[2]);
    int r = std::stoi(argv[3]);
    float sigma = std::stof(argv[4]);

    const int image_size = 28 * 28;
    const int num_images = n;
    
    std::string mnist_path = "~/Documents/Cholesky_GPU/src/MNIST_data/";
    std::string train_images_path = mnist_path + "train-images-idx3-ubyte";
    std::string train_labels_path = mnist_path + "train-labels-idx1-ubyte";
    std::string test_images_path = mnist_path + "t10k-images-idx3-ubyte";
    std::string test_labels_path = mnist_path + "t10k-labels-idx1-ubyte";

    // Expand tilde (~) manually if needed
    if (train_images_path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) {
            train_images_path.replace(0, 1, home);
        }
    }

    // Open MNIST image file
    std::ifstream file(train_images_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open MNIST image file!" << std::endl;
        exit(1);
    }

    // Skip 16-byte header
    file.seekg(16);

    // Allocate column-major matrix: image_size rows × num_images columns
    std::vector<float> mnist_matrix(image_size * num_images);

    std::cout << "loading in images" << std::endl;

    // Load and normalize pixels into column-major format
    for (int col = 0; col < num_images; ++col) {
        for (int row = 0; row < image_size; ++row) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            mnist_matrix[col * image_size + row] = static_cast<float>(pixel) / 255.0f;
            

        }
    }

    file.close();

    LegacyMatrix<float, int> Points(image_size, num_images, mnist_matrix.data(), image_size);
    


    std::vector<float> A_mat(n*k, static_cast<float>(0.0));
    LegacyMatrix<float, int> A(n, k, A_mat.data(), n);


    //make buffer G1, G2, G3
    std::vector<float> G1_mat(n*r, 0.0);
    LegacyMatrix<float, int> G1(n, r, G1_mat.data(), n);

    std::vector<float> G2_mat(r*r, static_cast<float>(0.0));
    LegacyMatrix<float, int> G2(r, r, G2_mat.data(), r);

    std::vector<float> G3_mat(r*r, 0.0);
    LegacyMatrix<float, int> G3(r, r, G3_mat.data(), r);

    //make buffer for diag_A
    std::vector<float> diag_A(n, 0.0);

    //scaling vec for fp8 A
    std::vector<float> A_exp_mat((int)(n/r)*(k/r), 0.0);

    std::set<int> S;
    rand_piv_RBF(A, A_exp_mat, G1, G2, G3, diag_A, Points, n, k, S, r, sigma);

    //print contents of S
    for(auto elem : S) {
        std::cout << elem << " ";
    }

    //now compute A(S, :)A(:, S) + lambda*N*A(S,S) and call CG_IR with it and A(S, :) y -> labels of training set
    int s = S.size();

    std::vector<float> A_S_mat(s*s, 0.0);
    LegacyMatrix<float, int> A_S(s, s, A_S_mat.data(), s);

    float lambda = 1.0/float(n);


    RBF_compressed(A_S, S, s, n ,lambda, 28*28, 1.0, Points);

    //now call CG on the labels. Get labels from MNIST file-
    // Open MNIST label file
    std::ifstream label_file(train_labels_path, std::ios::binary);
    if (!label_file) {
        std::cerr << "Failed to open MNIST label file!" << std::endl;
        exit(1);
    }

    // Skip 8-byte header (magic number + number of items)
    label_file.seekg(8);

    // Allocate label vector
    std::vector<unsigned char> labels(num_images);

    // Load labels
    label_file.read(reinterpret_cast<char*>(labels.data()), num_images);

    label_file.close();

    std::vector<float> labels_S;
    for (int idx : S) {
        labels_S.push_back(static_cast<float>(labels[idx]));
    }

    //transform labels by multiplying with A(:, S)
    transform_label(Points, labels_S, n, sigma, s);



    //now call CG_IR to compute beta
    using Matrix_type = LegacyMatrix<float, int>;
    using Vector_type = LegacyVector<float, int>;
    using MP_solve = CG_IR<Matrix_type, Vector_type, float, float>;

    double tol = 0.01;
    double conv_thresh = 0.0001;
    chol_mod chol_modif = chol_mod::GMW81;
    int block_size = 128

    MP_solve(A_S, labels_S, s, tol, conv_thresh, chol_modif, block_size);

    //now beta is stored in labels_S
    auto& beta = labels_S;



    // Evaluate training accuracy
    std::vector<float> predictions(num_images, 0.0f);
    int j = 0;
    for (int i = 0; i < num_images; i++) {
        float pred = 0.0f;
        int j = 0;
        for (int s_idx : S) {
            pred += rbf_value(mnist_matrix, i, s_idx, 1.0f, image_size) * labels_S[j];
            j++;
        }
        predictions[i] = pred;
    }

    int correct = 0;
    for (int i = 0; i < num_images; i++) {
        int pred_label = static_cast<int>(std::round(predictions[i]));
        int true_label = static_cast<int>(labels[i]);
        if (pred_label == true_label)
            correct++;
    }
    float accuracy = 100.0f * correct / num_images;
    std::cout << "Training set accuracy: " << accuracy << "%" << std::endl;

    // Load test labels
    int num_test_images = 1000;
    std::ifstream test_label_file(test_labels_path, std::ios::binary);
    if (!test_label_file) {
        std::cerr << "Failed to open MNIST test label file!" << std::endl;
        exit(1);
    }

    test_label_file.seekg(8); // Skip header
    std::vector<unsigned char> test_labels(num_test_images);
    test_label_file.read(reinterpret_cast<char*>(test_labels.data()), num_test_images);
    test_label_file.close();

    // Load test images
    std::ifstream test_file(test_images_path, std::ios::binary);
    if (!test_file) {
        std::cerr << "Failed to open MNIST test image file!" << std::endl;
        exit(1);
    }

    test_file.seekg(16); // Skip header
    std::vector<float> test_mnist_matrix(image_size * num_test_images);
    for (int col = 0; col < num_test_images; ++col) {
        for (int row = 0; row < image_size; ++row) {
            unsigned char pixel = 0;
            test_file.read(reinterpret_cast<char*>(&pixel), 1);
            test_mnist_matrix[col * image_size + row] = static_cast<float>(pixel) / 255.0f;
        }
    }
    test_file.close();

    // Evaluate test accuracy
    std::vector<float> test_predictions(num_test_images, 0.0f);
    for (int i = 0; i < num_test_images; i++) {
        float pred = 0.0f;
        int j = 0;
        for (int s_idx : S) {
            pred += rbf_value(test_mnist_matrix, i, s_idx, sigma, image_size) * labels_S[j];
            j++;
        }
        test_predictions[i] = pred;
    }

    int test_correct = 0;
    for (int i = 0; i < num_test_images; i++) {
        int pred_label = static_cast<int>(std::round(test_predictions[i]));
        int true_label = static_cast<int>(test_labels[i]);
        if (pred_label == true_label)
            test_correct++;
    }
    float test_accuracy = 100.0f * test_correct / num_test_images;
    std::cout << "Test set accuracy: " << test_accuracy << "%" << std::endl;

    std::cout << std::endl;

    return 0;
}
