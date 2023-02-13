#pragma once

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
                 const float _Complex alpha, _Bool conjA, const float _Complex *A, const int *sizeA, const int *outerSizeA, 
                 const float _Complex beta,                    float _Complex *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor);

void zTensorTranspose( const int *perm, const int dim,
                 const double _Complex alpha, _Bool conjA, const double _Complex *A, const int *sizeA, const int *outerSizeA, 
                 const double _Complex beta,                    double _Complex *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor);