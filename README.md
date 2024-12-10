# P3109-LAPACK
This repo contains some ongoing work for trying out the effectiveness of the new IEEE P3109 standard with some LAPACK algortihms.
We are currently looking at LU and Cholesky factorizations (to solve $Ax = b$) and QR factorization (to find $min||Ax - b||^2$). 
As is the case with many linear algebra mixed precision algorithms today, our appraoch is to perform a mixed precision decomposoition ($O(n^3)$ work), then perform a cheap iterative refinement in higher precision ($O(n^2)$). 

# Instructions on usage for Cholesky
First head into the directory Precimonious/GMRES-IR/scripts. Then, you can build the Choleksy example by just running make CG. This will produce an executable CG_IR in the same directory which will run your CG experiments.
To test the routine for different n and condition number, you can change 'n' and 'cond' in the file settings.json.
To try out different diagonal perturbation strategies, you can modify the 'chol_mod' field to be either NONE, GMW81, SE90 or SE99.
To test out different 8-bit precisions,
you can head into the "pivoted_cholesky.cpp" file and change all instances of 
"ml_dtypes::float8_ieee_p<p>" to "ml_dtypes::float8_ieee_p<x>", where x is the precision you want to test out with (x - 1 is the number of mantissa bits).

