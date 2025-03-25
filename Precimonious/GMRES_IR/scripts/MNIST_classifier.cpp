#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include "rbf_kernel.hpp"    // Implements RBF kernel matrix
#include "rp_cholesky.hpp"   // Implements RPCholesky(A, k) and returns F, S
#include "cg_solver.hpp"     // Assumed header for CG_IR

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

template<typename Matrix, typename Vector, typename Scalar>
void gemv(Matrix A, Vector x, Scalar alpha, Scalar beta, Vector b, int n, int m)
{
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            b[i] += alpha*A(i,j) * x[j];
        }
        b[i] += beta*b[i];
    }
}

// Loads MNIST image or label data from a binary file
bool load_mnist_bytes(const std::string& image_path, const std::string& label_path, int num_images, int image_size,
                      Matrix& X, Vector& y, int target_label) {
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(label_path, std::ios::binary);
    if (!image_file.is_open() || !label_file.is_open()) {
        std::cerr << "Failed to open MNIST files.\n";
        return false;
    }

    // Skip headers
    image_file.ignore(16);
    label_file.ignore(8);

    X.resize(image_size, num_images);
    y.resize(num_images);

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel;
            image_file.read((char*)&pixel, 1);
            X(j, i) = pixel / 255.0f;
        }
        unsigned char label;
        label_file.read((char*)&label, 1);
        y(i) = (label == target_label) ? 1.0f : 0.0f;
    }
    return true;
}

// Accelerated Kernel Ridge Regression
template<typename Matrix, typename Vector, typename Points, typename piv_t>
void accelerated_kernel_ridge(
    Matrix& X,
    Matrix& X_S,
    Matrix& XS_,
    Matrix& XSS,
    Vector& diag_X,
    Vector& y,
    Points& training_set,
    piv_t& left_piv,
    piv_t& right_piv,
    int n,
    int k,
    float lambda,
    Vector& alpha_out,
    std::set<int>& pivots_out
) {


    pivots_out = rand_piv_RBF(X, diag_X, training_set, right_piv, left_piv, k);

    
    for( auto s : pivots_out) {
        int col_cnt = 0;
        for(int i = 0; i < n; i++) {
            X_S(i, col_cnt) = X(i, s);
            XS_(col_cnt, i) = X(s, i);
            col_cnt++;
        }
        
    }
    int x_ind = 0; int y_ind = 0;
    for(auto s1 : pivots_out) {
        y_ind = 0;
        for(auto s2 : pivots_out) {
            XSS(x_ind, y_ind) = X(s1, s2);
            y_ind++;
        }
        x_ind++;
    }

    

    
    block_gemm(X_S, XS_, XSS, X_S, XS_);

    gemv(X_S, y, 1.0, 0.0, y, pivots_out.size(), n);

    // Solve (ASS + lambda * I) * x = AS^T * y using CG_IR
    CG_IR(system_matrix, rhs, middle_solution, 1e-8, 1e-10);  // eps_prime and tol

    alpha_out = AS * middle_solution;
}

int main() {
    Matrix train_X, test_X, val_X;
    Vector train_y, test_y, val_y;

    int image_size = 28 * 28;
    int train_count = 60000;
    int test_count = 10000;
    int val_count = 10000; // If needed
    int target_digit = 3;

    load_mnist_bytes("/home/eecs/sudhanvakulkarni/mixed_prec_MNIST/mnist_data/train-images.idx3-ubyte", "/home/eecs/sudhanvakulkarni/mixed_prec_MNIST/mnist_data/train-labels.idx1-ubyte", train_count, image_size, train_X, train_y, target_digit);
    load_mnist_bytes("/home/eecs/sudhanvakulkarni/mixed_prec_MNIST/mnist_data/t10k-images.idx3-ubyte", "/home/eecs/sudhanvakulkarni/mixed_prec_MNIST/mnist_data/t10k-labels.idx1-ubyte", test_count, image_size, test_X, test_y, target_digit);

    // Center and normalize
    train_X = train_X.colwise() - train_X.rowwise().mean();
    train_X /= train_X.norm();

    int k = 300;      // approximation rank
    float lambda = 1e-3;
    Vector alpha;
    std::vector<int> pivots;

    // Train
    accelerated_kernel_ridge(train_X, train_y, k, lambda, alpha, pivots);

    // Predict on test data
    Matrix K_test = compute_rbf_kernel(test_X, train_X(Eigen::all, pivots));
    Vector pred = K_test * alpha;

    // Evaluate
    int correct = 0;
    for (int i = 0; i < pred.size(); ++i) {
        int predicted = pred[i] > 0.5 ? 1 : 0;
        int actual = test_y[i] > 0.5 ? 1 : 0;
        if (predicted == actual) correct++;
    }
    std::cout << "Accuracy: " << 100.0 * correct / pred.size() << "%\n";
    return 0;
}
