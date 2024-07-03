#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <inttypes.h>
#include <time.h>
#include "float8.h"
#include <stdio.h>
#include <iostream>
#define ITERS 10


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

int main() {
  
  typedef ml_dtypes::float8_ieee_p<4> float8_ieee_p4;
  /****** BEGIN PRECIMONIOUOS PREAMBLE ******/
  // variables for logging/checking
  double epsilon = 1.0e-4;
  int l;

  // variables for timing measurement
  clock_t start_time, end_time;  
  double cpu_time_used;

  start_time = clock();
  

  /****** END PRECIMONIOUOS PREAMBLE ******/
  
  int i, j, k, n = 1000000;
  float8_ieee_p4 h;
  float8_ieee_p4  t1;
  float8_ieee_p4  t2;
  float8_ieee_p4  dppi;
  float8_ieee_p4  s1;
  

  
  
  for (l = 0; l < ITERS; l++) {
    t1 = -1.0;
    dppi = acos(t1);
    s1 = 0.0;
    t1 = 0.0;
    h = dppi / n;

    for( i = 1; i <= n; i++ ) {
      t2 = fun (static_cast<float8_ieee_p4>(i) * h);
      s1 = s1 + sqrt (h*h + (t2 - t1)*(t2 - t1));
      t1 = t2;
    }
  }



  /***** BEGIN PRECIMONIOUS ACCURACY CHECKING AND LOGGING ******/

  // record in file verified, s1, err
  double zeta_verify_value = 5.7957763224130E+00;
  double err;
  bool verified;
  err = static_cast<double>(fabs(s1 - zeta_verify_value)) / zeta_verify_value;
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
  cpu_time_used = ((double) (start_time - start_time)) / CLOCKS_PER_SEC;
  FILE *fp_t = fopen("./time.txt", "w");
  fprintf(fp_t, "%f\n", cpu_time_used);


  /****** END PRECIMONIOUS ACCURACY CHECKING AND LOGGING ******/

  return 0;
}

