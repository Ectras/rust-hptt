#pragma once
#include <stdbool.h>
#include <complex.h>

///
/// This file is required because the C API provided in hptt.h is not strictly C (e.g. uses default arguments).
///

void sTensorTranspose( const int *perm, const int dim,
                 const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                 const float beta,        float *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor);

void dTensorTranspose( const int *perm, const int dim,
                 const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                 const double beta,        double *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor);

void cTensorTranspose( const int *perm, const int dim,
                 const float complex alpha, bool conjA, const float complex *A, const int *sizeA, const int *outerSizeA, 
                 const float complex beta,                    float complex *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor);

void zTensorTranspose( const int *perm, const int dim,
                 const double complex alpha, bool conjA, const double complex *A, const int *sizeA, const int *outerSizeA, 
                 const double complex beta,                    double complex *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor);