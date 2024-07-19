# P3109-LAPACK
This repo contains some ongoing work for trying out the effectiveness of the new IEEE P3109 standard with some LAPACK algortihms.
We are currently looking at LU factorization (to solve $Ax = b$) and QR factorization (to find $min||Ax - b||^2$). 
As is the case with many linear algebra mixed precision algorithms today, our appraoch is perform a mixed precision decomposoition ($O(n^3)$ work), then perform a cheap iterative refinement in higher precision ($O(n^2)$).
