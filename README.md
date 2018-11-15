# ILT
Mathematical Library for 1D and 2D Inverse Laplace Transform

This Library is able to rapidly compute an estimate of the Inverse Laplace Transform of a data-set accquired on regular or irregular grid.

It uses the Nonnegativity-constrained least squares code from J. Kim and H. Park, found in nnls.py (see details and reference therein).

## ILT_1D.py is for 1D data-sets.
Given a a set of $N$ experimental points $E_n$, sampling at time $T_n$ the evolution of a damping signal, following the Laplace law:
$$ E_n = \sum_{m=1}^M exp(-S_m T_n) $$
it solves the Laplace problem and computes an estimate $\hat{S}$ of the Laplace spectrum $S$, assuming the positivity of the coefficients, by minimizing the least square estimate:
$$ LS = \sum_{n=1}^N \left( E_n - \sum_{m=1}^M exp(-\hat{S}_m T_n) \right)^2 $$
It is is a direct application of nnls.py

ILT_2D.py is for 2D data-sets

## License
This code is released under the GPLv3 license.