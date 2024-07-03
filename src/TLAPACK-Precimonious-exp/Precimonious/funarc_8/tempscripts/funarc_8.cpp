#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <inttypes.h>
#include <time.h>
#include "float8.h"
#include "../eigen/Eigen/Core"
#include <stdio.h>
#include <iostream>
#define ITERS 1



using namespace ml_dtypes;
using namespace Eigen;

template <typename T>
T sqrt(T x) {
  return static_cast<T>(std::sqrt(static_cast<double>(x)));
}

template <typename T>
T acos(T x) {
  return static_cast<T>(std::acos(static_cast<double>(x)));
}

template <typename T>
T sin(T x) {
  return static_cast<T>(std::acos(static_cast<double>(x)));
}

template <typename T>
T fabs(T x) {
  return static_cast<T>(std::fabs(static_cast<double>(x)));
}

template <typename T>
T fun( T x ) {
  int k, n = 5;
  T t1 = 1.0L;
  T d1 = 1.0L;

  t1 = x;

  for( k = 1; k <= n; k++ )
  {
    d1 = static_cast<T>(2.0) * d1;
    t1 = t1 + sin (d1 * x) / d1;
  }

  return t1;
}

template <typename T1>
void measure_time(T1 h, double& cpu_time_used) {
  cpu_time_used = cpu_time_used + static_cast<double>(0.0);
}

void measure_time(float h, double& cpu_time_used) {
  cpu_time_used = cpu_time_used + static_cast<double>(4.0);
}

void measure_time(double h, double& cpu_time_used) {
  cpu_time_used = cpu_time_used + static_cast<double>(8.0);
}

template<int p>
void measure_time(ml_dtypes::float8_ieee_p<p> h, double& cpu_time_used) {
  cpu_time_used = cpu_time_used + static_cast<double>(1.0);
}

void measure_time(Eigen::bfloat16 h, double& cpu_time_used) {
  cpu_time_used = cpu_time_used + static_cast<double>(2.0);
}

void measure_time(Eigen::half h, double& cpu_time_used) {
  cpu_time_used = cpu_time_used + static_cast<double>(2.0);
}

void measure_time(long double h, double& cpu_time_used) {
  cpu_time_used = cpu_time_used + static_cast<double>(16.0);
}


template <typename T1, typename T2, typename T3, typename T4, typename T5>
void fun_a(T2& t1, T3& t2, T4& dppi, T5& s1) {
  int l;
  int i, j, k, n = 1000000;
  T1 h;
 
  for (l = 0; l < ITERS; l++) {
    t1 = static_cast<T2>(-1.0);
    dppi = acos(static_cast<T4>(t1));
    s1 = static_cast<T5>(0.0);                          //optimal is binary8p3
    t1 = static_cast<T2>(0.0);                          //optimal is binary8p3
    h = static_cast<T1>(dppi / static_cast<T4>(n));     //optimal data type is fp32

    for( i = 1; i <= n; i++ ) {
      t2 = static_cast<T3>(fun (static_cast<T1>(i) * h));   //optimal is binary8p3
      s1 = s1 + sqrt (static_cast<T5>(h*h) + static_cast<T5>((static_cast<T2>(t2) - t1)*(static_cast<T2>(t2) - t1)));
      s1 = s1 + static_cast<T5>(sqrt(static_cast<T2>(h*h) + (static_cast<T2>(t2) - t1)*(static_cast<T2>(t2) - t1))); 

      
      
      t1 = static_cast<T2>(t2);
    }
  }

}

int main() {
  
  typedef ml_dtypes::float8_ieee_p<4> float8_ieee_p4;
  typedef ml_dtypes::float8_ieee_p<3> float8_ieee_p3;
  /****** BEGIN PRECIMONIOUOS PREAMBLE ******/
  // variables for logging/checking
  double epsilon = 10.0;
  int l;

  // variables for timing measurement
  clock_t start_time, end_time;  
  double cpu_time_used = 0.0;

  start_time = clock();
  

  /****** END PRECIMONIOUOS PREAMBLE ******/
  
  float8_ieee_p3  t1;
  float8_ieee_p3  t2;
  float8_ieee_p3  dppi;
  float8_ieee_p3  s1;
  std::vector<float8_ieee_p3> v1;

  fun_a<double, float8_ieee_p3 ,float8_ieee_p3 ,float8_ieee_p3 ,float8_ieee_p3 >(t1, t2, dppi, s1);
  

  /***** BEGIN PRECIMONIOUS ACCURACY CHECKING AND LOGGING ******/

  // record in file verified, s1, err
  double zeta_verify_value = 5.7957763224130E+00;
  double err;
  bool verified;
  err = static_cast<double>(fabs(static_cast<double>(s1) - zeta_verify_value)) / zeta_verify_value;
  if (err <= epsilon) {
      verified = true;
      std::cout << " VERIFICATION SUCCESSFUL\n";
      std::cout << " Zeta is " << s1 << std::endl;
      std::cout << " Error is " <<  err << std::endl;
  } else {
      verified = false;
      std::cout << " VERIFICATION FAILED\n";
      std::cout << " Zeta is " << s1 << std::endl;
      std::cout << " Error is " <<  err << std::endl;
  }
  FILE *fp = fopen("./log.txt", "w");
  fputs(verified ? "true\n" : "false\n", fp);
  fprintf(fp, "%20.13E\n", static_cast<double>(s1));
  fprintf(fp, "%20.13E\n", static_cast<double>(err));

  // record time
  end_time = clock(); 
  FILE *fp_t = fopen("./time.txt", "w");
  measure_time(t1, cpu_time_used);
  measure_time(t2, cpu_time_used);
  measure_time(dppi, cpu_time_used);
  measure_time(s1, cpu_time_used);
  fprintf(fp_t, "%f\n", cpu_time_used);


  /****** END PRECIMONIOUS ACCURACY CHECKING AND LOGGING ******/

  return 0;
}

