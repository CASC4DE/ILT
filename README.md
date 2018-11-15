# ILT
Mathematical Library for 1D and 2D Inverse Laplace Transform

This Library is able to rapidly compute an estimate of the Inverse Laplace Transform of a data-set accquired on regular or irregular grid.

It uses the Nonnegativity-constrained least squares code from J. Kim and H. Park, found in `nnls.py` (see details and reference therein).

## ILT_1D.py is for 1D data-sets.
Given a a set of $N$ experimental points $E_n$, sampling at time $T_n$ the evolution of a damping signal, following the Laplace law:

$$ E_n = \sum_{m=1}^M exp(-S_m T_n) $$

it solves the Laplace problem and computes an estimate $\hat{S}$ of the Laplace spectrum $S$, assuming the positivity of the coefficients, by minimizing the least square estimate:

$$ LS = \sum_{n=1}^N \left( E_n - \sum_{m=1}^M exp(-\hat{S}_m T_n) \right)^2 $$

It is is a direct application of `nnls.py`

## ILT_2D.py is for 2D data-sets.
This code solves the 2D problem equivalent to the 1D problem above.
The code uses the idea from Song et al of reducing the complexity of the 2D experimental matrix.
The method of truncated SVD is implmented as well as a faster approach based on random projection.

## Caveat
This code is part of an ongoing project.
It contains many parts not fully tested, or even non-functional.
The sequence presented in the two main entries: `ILT_1D.py` and `ILT_2D.py` is functionnal though.
This program has been develped for internal usage in CASC4DE.
There is no warranty whatsoever for this program to actually execute correctly what it is meant for. 


## License
This code is provided in a Free Open-Source form by CASC4DE: www.casc4de.eu

All inquiries about this code should be sent to contact@casc4de.eu

This code is released under the GPLv3 license.

## Authors
This program was mostly developped by

- Lionel Chiron: lionel.chiron@gmail.com / lionel.chiron@casc4de.eu
- Marc-André Delsuc: mad@delsuc.net
- Laura Duciel: laura.duciel@casc4de.eu

with the help of

- Cammile Marin-Beluffi: camille.Beluffi@casc4de.eu

