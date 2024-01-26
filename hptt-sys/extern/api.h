#pragma once

///
/// This file is required because the C API provided in hptt.h is not strictly C (e.g. uses default arguments).
///

/**
 * \brief Computes the out-of-place tensor transposition of A into B
 *
 * A tensor transposition plan is a data structure that encodes the execution of the tensor transposition.
 * HPTT supports tensor transpositions of the form: 
 * B_{π(i₀,i₁,...)} = α * A_{i₀,i₁,...} + β * B_{π(i₀,i₁,...)}.
 * The plan can be reused over several transpositions.
 *
 * \param[in] perm dim-dimensional array representing the permutation of the indices.
 *  For instance, `perm = [1,0,2]` denotes the following transposition: `B_{i₁,i₀,i₂} <- A_{i₀,i₁,i₂}`.
 * \param[in] dim Dimensionality of the tensors
 * \param[in] alpha scaling factor for A
 * \param[in] A Pointer to the raw-data of the input tensor A
 * \param[in] sizeA dim-dimensional array that stores the sizes of each dimension of A 
 * \param[in] outerSizeA dim-dimensional array that stores the outer-sizes of each dimension of A.
 *  This parameter may be NULL, indicating that the outer-size is equal to sizeA.
 *  If outerSizeA is not NULL, outerSizeA[i] >= sizeA[i] for all 0 <= i < dim must hold.
 *  This option enables HPTT to operate on sub-tensors.
 * \param[in] beta scaling factor for B
 * \param[in,out] B Pointer to the raw-data of the output tensor B
 * \param[in] outerSizeB dim-dimensional array that stores the outer-sizes of each dimension of B.
 *  This parameter may be NULL, indicating that the outer-size is equal to the perm(sizeA). 
 *  If outerSizeA is not NULL, outerSizeB[i] >= perm(sizeA)[i] for all 0 <= i < dim must hold.
 *  This option enables HPTT to operate on sub-tensors.
 * \param[in] numThreads number of threads that participate in this tensor transposition.
 * \param[in] useRowMajor This flag indicates whether a row-major memory layout should be used (default: off = column-major).
 */
void sTensorTranspose( const int *perm, const int dim,
                 const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                 const float beta,        float *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor);

/**
 * \brief Computes the out-of-place tensor transposition of A into B
 *
 * A tensor transposition plan is a data structure that encodes the execution of the tensor transposition.
 * HPTT supports tensor transpositions of the form: 
 * B_{π(i₀,i₁,...)} = α * A_{i₀,i₁,...} + β * B_{π(i₀,i₁,...)}.
 * The plan can be reused over several transpositions.
 *
 * \param[in] perm dim-dimensional array representing the permutation of the indices.
 *  For instance, `perm = [1,0,2]` denotes the following transposition: `B_{i₁,i₀,i₂} <- A_{i₀,i₁,i₂}`.
 * \param[in] dim Dimensionality of the tensors
 * \param[in] alpha scaling factor for A
 * \param[in] A Pointer to the raw-data of the input tensor A
 * \param[in] sizeA dim-dimensional array that stores the sizes of each dimension of A 
 * \param[in] outerSizeA dim-dimensional array that stores the outer-sizes of each dimension of A.
 *  This parameter may be NULL, indicating that the outer-size is equal to sizeA.
 *  If outerSizeA is not NULL, outerSizeA[i] >= sizeA[i] for all 0 <= i < dim must hold.
 *  This option enables HPTT to operate on sub-tensors.
 * \param[in] beta scaling factor for B
 * \param[in,out] B Pointer to the raw-data of the output tensor B
 * \param[in] outerSizeB dim-dimensional array that stores the outer-sizes of each dimension of B.
 *  This parameter may be NULL, indicating that the outer-size is equal to the perm(sizeA). 
 *  If outerSizeA is not NULL, outerSizeB[i] >= perm(sizeA)[i] for all 0 <= i < dim must hold.
 *  This option enables HPTT to operate on sub-tensors.
 * \param[in] numThreads number of threads that participate in this tensor transposition.
 * \param[in] useRowMajor This flag indicates whether a row-major memory layout should be used (default: off = column-major).
 */
void dTensorTranspose( const int *perm, const int dim,
                 const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                 const double beta,        double *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor);

/**
 * \brief Computes the out-of-place tensor transposition of A into B
 *
 * A tensor transposition plan is a data structure that encodes the execution of the tensor transposition.
 * HPTT supports tensor transpositions of the form: 
 * B_{π(i₀,i₁,...)} = α * A_{i₀,i₁,...} + β * B_{π(i₀,i₁,...)}.
 * The plan can be reused over several transpositions.
 *
 * \param[in] perm dim-dimensional array representing the permutation of the indices.
 *  For instance, `perm = [1,0,2]` denotes the following transposition: `B_{i₁,i₀,i₂} <- A_{i₀,i₁,i₂}`.
 * \param[in] dim Dimensionality of the tensors
 * \param[in] alpha scaling factor for A
 * \param[in] A Pointer to the raw-data of the input tensor A
 * \param[in] sizeA dim-dimensional array that stores the sizes of each dimension of A 
 * \param[in] outerSizeA dim-dimensional array that stores the outer-sizes of each dimension of A.
 *  This parameter may be NULL, indicating that the outer-size is equal to sizeA.
 *  If outerSizeA is not NULL, outerSizeA[i] >= sizeA[i] for all 0 <= i < dim must hold.
 *  This option enables HPTT to operate on sub-tensors.
 * \param[in] beta scaling factor for B
 * \param[in,out] B Pointer to the raw-data of the output tensor B
 * \param[in] outerSizeB dim-dimensional array that stores the outer-sizes of each dimension of B.
 *  This parameter may be NULL, indicating that the outer-size is equal to the perm(sizeA). 
 *  If outerSizeA is not NULL, outerSizeB[i] >= perm(sizeA)[i] for all 0 <= i < dim must hold.
 *  This option enables HPTT to operate on sub-tensors.
 * \param[in] numThreads number of threads that participate in this tensor transposition.
 * \param[in] useRowMajor This flag indicates whether a row-major memory layout should be used (default: off = column-major).
 */
void cTensorTranspose( const int *perm, const int dim,
                 const float _Complex alpha, _Bool conjA, const float _Complex *A, const int *sizeA, const int *outerSizeA, 
                 const float _Complex beta,                    float _Complex *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor);

/**
 * \brief Computes the out-of-place tensor transposition of A into B
 *
 * A tensor transposition plan is a data structure that encodes the execution of the tensor transposition.
 * HPTT supports tensor transpositions of the form: 
 * B_{π(i₀,i₁,...)} = α * A_{i₀,i₁,...} + β * B_{π(i₀,i₁,...)}.
 * The plan can be reused over several transpositions.
 *
 * \param[in] perm dim-dimensional array representing the permutation of the indices.
 *  For instance, `perm = [1,0,2]` denotes the following transposition: `B_{i₁,i₀,i₂} <- A_{i₀,i₁,i₂}`.
 * \param[in] dim Dimensionality of the tensors
 * \param[in] alpha scaling factor for A
 * \param[in] A Pointer to the raw-data of the input tensor A
 * \param[in] sizeA dim-dimensional array that stores the sizes of each dimension of A 
 * \param[in] outerSizeA dim-dimensional array that stores the outer-sizes of each dimension of A.
 *  This parameter may be NULL, indicating that the outer-size is equal to sizeA.
 *  If outerSizeA is not NULL, outerSizeA[i] >= sizeA[i] for all 0 <= i < dim must hold.
 *  This option enables HPTT to operate on sub-tensors.
 * \param[in] beta scaling factor for B
 * \param[in,out] B Pointer to the raw-data of the output tensor B
 * \param[in] outerSizeB dim-dimensional array that stores the outer-sizes of each dimension of B.
 *  This parameter may be NULL, indicating that the outer-size is equal to the perm(sizeA). 
 *  If outerSizeA is not NULL, outerSizeB[i] >= perm(sizeA)[i] for all 0 <= i < dim must hold.
 *  This option enables HPTT to operate on sub-tensors.
 * \param[in] numThreads number of threads that participate in this tensor transposition.
 * \param[in] useRowMajor This flag indicates whether a row-major memory layout should be used (default: off = column-major).
 */
void zTensorTranspose( const int *perm, const int dim,
                 const double _Complex alpha, _Bool conjA, const double _Complex *A, const int *sizeA, const int *outerSizeA, 
                 const double _Complex beta,                    double _Complex *B,                   const int *outerSizeB, 
                 const int numThreads, const int useRowMajor);