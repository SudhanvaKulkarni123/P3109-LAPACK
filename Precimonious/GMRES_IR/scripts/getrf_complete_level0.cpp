///@author Sudhanva Kulkarni, UC Berkeley
///this file contains level-1 BLAS code for finding LU factroization of a matrix with complete pivoting.
///So it returns A = PLUQ

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/ger.hpp"

