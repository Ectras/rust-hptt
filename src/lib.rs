mod hptt {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

static mut DEFAULT_NUM_THREADS: u32 = 1;
static mut USE_ROW_MAJOR: bool = false;

mod implementations {
    use std::mem::transmute;

    use num_complex::{Complex32, Complex64};

    use crate::{
        hptt::{
            __BindgenComplex, cTensorTranspose, dTensorTranspose, sTensorTranspose,
            zTensorTranspose,
        },
        DEFAULT_NUM_THREADS, USE_ROW_MAJOR,
    };

    pub trait Transposable<T> {
        fn transpose(
            perm: &[i32],
            alpha: T,
            a: &[T],
            size_a: &[i32],
            outer_size_a: Option<&[i32]>,
            beta: T,
            b: Option<Vec<T>>,
            outer_size_b: Option<&[i32]>,
            num_threads: u32,
            use_row_major: bool,
        ) -> Vec<T>;

        fn transpose_simple(perm: &[i32], a: &[T], size_a: &[i32]) -> Vec<T>;
    }

    /// Returns a vector with the requested capacity. If vec is given, it should either
    /// have that many elements or be empty, in which case its capacity will be set accordingly.
    fn with_capacity<T>(vec: Option<Vec<T>>, capacity: usize) -> Vec<T> {
        // Get a vector with enough capacity
        if let Some(mut v) = vec {
            if v.len() >= capacity {
                // Given output is fully initialized, can be used directly
                v
            } else if v.is_empty() {
                // Given output is empty, make sure the capacity suffices
                v.reserve_exact(capacity - v.capacity());
                v
            } else {
                panic!("Output vector must either have same length as input, or must be empty");
            }
        } else {
            // Create a new vector
            Vec::with_capacity(capacity)
        }
    }

    impl Transposable<f32> for () {
        fn transpose(
            perm: &[i32],
            alpha: f32,
            a: &[f32],
            size_a: &[i32],
            outer_size_a: Option<&[i32]>,
            beta: f32,
            b: Option<Vec<f32>>,
            outer_size_b: Option<&[i32]>,
            num_threads: u32,
            use_row_major: bool,
        ) -> Vec<f32> {
            assert!(
                matches!(outer_size_a, None) && matches!(outer_size_b, None),
                "Outer size not supported yet"
            );

            let mut out = with_capacity(b, a.len());
            unsafe {
                sTensorTranspose(
                    perm.as_ptr(),
                    perm.len().try_into().unwrap(),
                    alpha,
                    a.as_ptr(),
                    size_a.as_ptr(),
                    outer_size_a.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    beta,
                    out.as_mut_ptr(),
                    outer_size_b.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    num_threads.try_into().unwrap(),
                    use_row_major.try_into().unwrap(),
                );
                out.set_len(a.len());
            }
            out
        }

        fn transpose_simple(perm: &[i32], a: &[f32], size_a: &[i32]) -> Vec<f32> {
            Self::transpose(
                perm,
                1.0f32,
                a,
                size_a,
                None,
                0.0f32,
                None,
                None,
                unsafe { DEFAULT_NUM_THREADS },
                unsafe { USE_ROW_MAJOR },
            )
        }
    }

    impl Transposable<f64> for () {
        fn transpose(
            perm: &[i32],
            alpha: f64,
            a: &[f64],
            size_a: &[i32],
            outer_size_a: Option<&[i32]>,
            beta: f64,
            b: Option<Vec<f64>>,
            outer_size_b: Option<&[i32]>,
            num_threads: u32,
            use_row_major: bool,
        ) -> Vec<f64> {
            assert!(
                matches!(outer_size_a, None) && matches!(outer_size_b, None),
                "Outer size not supported yet"
            );
            let mut out = with_capacity(b, a.len());
            unsafe {
                dTensorTranspose(
                    perm.as_ptr(),
                    perm.len().try_into().unwrap(),
                    alpha,
                    a.as_ptr(),
                    size_a.as_ptr(),
                    outer_size_a.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    beta,
                    out.as_mut_ptr(),
                    outer_size_b.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    num_threads.try_into().unwrap(),
                    use_row_major.try_into().unwrap(),
                );
                out.set_len(a.len());
            }
            out
        }

        fn transpose_simple(perm: &[i32], a: &[f64], size_a: &[i32]) -> Vec<f64> {
            Self::transpose(
                perm,
                1.0,
                a,
                size_a,
                None,
                0.0,
                None,
                None,
                unsafe { DEFAULT_NUM_THREADS },
                unsafe { USE_ROW_MAJOR },
            )
        }
    }

    impl Transposable<Complex32> for () {
        fn transpose(
            perm: &[i32],
            alpha: Complex32,
            a: &[Complex32],
            size_a: &[i32],
            outer_size_a: Option<&[i32]>,
            beta: Complex32,
            b: Option<Vec<Complex32>>,
            outer_size_b: Option<&[i32]>,
            num_threads: u32,
            use_row_major: bool,
        ) -> Vec<Complex32> {
            assert!(
                matches!(outer_size_a, None) && matches!(outer_size_b, None),
                "Outer size not supported yet"
            );
            let mut out: Vec<Complex32> = with_capacity(b, a.len());
            unsafe {
                cTensorTranspose(
                    perm.as_ptr(),
                    perm.len().try_into().unwrap(),
                    transmute(alpha),
                    false,
                    a.as_ptr().cast::<__BindgenComplex<f32>>(),
                    size_a.as_ptr(),
                    outer_size_a.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    transmute(beta),
                    out.as_mut_ptr().cast::<__BindgenComplex<f32>>(),
                    outer_size_b.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    num_threads.try_into().unwrap(),
                    use_row_major.try_into().unwrap(),
                );
                out.set_len(a.len());
            }
            out
        }

        fn transpose_simple(perm: &[i32], a: &[Complex32], size_a: &[i32]) -> Vec<Complex32> {
            Self::transpose(
                perm,
                Complex32::new(1.0f32, 0.0f32),
                a,
                size_a,
                None,
                Complex32::new(0.0f32, 0.0f32),
                None,
                None,
                unsafe { DEFAULT_NUM_THREADS },
                unsafe { USE_ROW_MAJOR },
            )
        }
    }

    impl Transposable<Complex64> for () {
        fn transpose(
            perm: &[i32],
            alpha: Complex64,
            a: &[Complex64],
            size_a: &[i32],
            outer_size_a: Option<&[i32]>,
            beta: Complex64,
            b: Option<Vec<Complex64>>,
            outer_size_b: Option<&[i32]>,
            num_threads: u32,
            use_row_major: bool,
        ) -> Vec<Complex64> {
            assert!(
                matches!(outer_size_a, None) && matches!(outer_size_b, None),
                "Outer size not supported yet"
            );
            let mut out: Vec<Complex64> = with_capacity(b, a.len());
            unsafe {
                zTensorTranspose(
                    perm.as_ptr(),
                    perm.len().try_into().unwrap(),
                    transmute(alpha),
                    false,
                    a.as_ptr().cast::<__BindgenComplex<f64>>(),
                    size_a.as_ptr(),
                    outer_size_a.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    transmute(beta),
                    out.as_mut_ptr().cast::<__BindgenComplex<f64>>(),
                    outer_size_b.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    num_threads.try_into().unwrap(),
                    use_row_major.try_into().unwrap(),
                );
                out.set_len(a.len());
            }
            out
        }

        fn transpose_simple(perm: &[i32], a: &[Complex64], size_a: &[i32]) -> Vec<Complex64> {
            Self::transpose(
                perm,
                Complex64::new(1.0, 0.0),
                a,
                size_a,
                None,
                Complex64::new(0.0, 0.0),
                None,
                None,
                unsafe { DEFAULT_NUM_THREADS },
                unsafe { USE_ROW_MAJOR },
            )
        }
    }
}

/// Computes B_{\pi(i_0,i_1,...)} = \alpha * A_{i_0,i_1,...} + \beta * B_{\pi(i_0,i_1,...)}.
pub fn transpose<T>(
    perm: &[i32],
    alpha: T,
    a: &[T],
    size_a: &[i32],
    beta: T,
    b: Option<Vec<T>>,
    num_threads: u32,
    use_row_major: bool,
) -> Vec<T>
where
    (): implementations::Transposable<T>,
{
    <() as implementations::Transposable<T>>::transpose(
        perm,
        alpha,
        a,
        size_a,
        None,
        beta,
        b,
        None,
        num_threads,
        use_row_major,
    )
}

/// Transposes the data in `a`, i.e. permutes the axes in the specified way.
/// It uses the global thread and row/column major settings.
///
/// # Example
/// ```
/// # use hptt_sys::transpose_simple;
/// let a = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // flat data
/// let shape = &[2, 3]; // actual shape of 'a' (2x3 matrix)
/// let perm = &[1, 0]; // swap the axes: put axis 1 first, axis 0 second
/// let b = transpose_simple(perm, a, shape);
/// // 'b' is now the flat data of a (3x2) matrix
/// assert_eq!(b[0], a[0]);
/// assert_eq!(b[1], a[2]);
/// assert_eq!(b[2], a[4]);
/// // ...
/// ```
pub fn transpose_simple<T>(perm: &[i32], a: &[T], size_a: &[i32]) -> Vec<T>
where
    (): implementations::Transposable<T>,
{
    <() as implementations::Transposable<T>>::transpose_simple(perm, a, size_a)
}

/// Creates the permuted version of an array. Can be used to compute the new shape after
/// transposing an array.
///
/// # Example
/// ```
/// # use hptt_sys::permute;
/// let arr = &[2, 4, 3, 1];
/// let perm = &[3, 2, 0, 1];
/// let out = permute(perm, arr);
/// assert_eq!(out, vec![1, 3, 2, 4]);
/// ```
pub fn permute<T>(perm: &[i32], arr: &[T]) -> Vec<T>
where
    T: Copy,
{
    (0..arr.len()).map(|i| arr[perm[i] as usize]).collect()
}

/// Creates the inverse permuted version of an array. Used to access values in transposed array.
///
/// # Example
/// ```
/// # use hptt_sys::permute;
/// # use hptt_sys::inv_permute;
/// let arr = &[2, 4, 3, 1];
/// let perm = &[3, 2, 0, 1];
/// let out = permute(perm, arr);
/// let perm2 = inv_permute(perm, &out);
/// assert_eq!(perm2, vec![2, 4, 3, 1]);
/// ```
pub fn inv_permute<T>(perm: &[i32], arr: &[T]) -> Vec<T>
where
    T: Copy,
{
    let mut indices = (0..perm.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &perm[i]);
    (0..arr.len()).map(|i| arr[indices[i]]).collect()
}

/// Sets the number of threads used by [`transpose_simple<T>()`]. This is a global property,
/// shared across threads but not guarded by a mutex. Hence, race conditions can happen.
pub fn set_number_of_threads(threads: u32) {
    unsafe {
        DEFAULT_NUM_THREADS = threads;
    }
}

/// Sets whether to use row or column major for [`transpose_simple<T>()`]. This is a global property,
/// shared across threads but not guarded by a mutex. Hence, race conditions can happen.
pub fn set_use_row_major(row_major: bool) {
    unsafe {
        USE_ROW_MAJOR = row_major;
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use num_complex::Complex64;

    use super::*;

    fn test_transposed<T>(original: &[T], transposed: &[T], permutated_indices: &[usize])
    where
        T: Debug + PartialEq,
    {
        for (i, j) in permutated_indices.iter().enumerate() {
            assert_eq!(transposed[i], original[*j]);
        }
    }

    #[test]
    fn f64_tensor() {
        let a = &[
            0.1, 0.65, 0.34, 0.76, 0.54, 0.17, 0.0, 0.63, 0.37, 0.22, 0.05, 0.17,
        ];

        let b = transpose(&[3, 2, 0, 1], 1.0, a, &[2, 2, 3, 1], 0.0, None, 1, true);

        test_transposed(a, &b, &[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]);
    }

    #[test]
    fn f64_tensor_simple() {
        // transpose_simple uses default column-major setting
        let a = &[
            0.1, 0.65, 0.34, 0.76, 0.54, 0.17, 0.0, 0.63, 0.37, 0.22, 0.05, 0.17,
        ];

        let b = transpose_simple(&[3, 2, 0, 1], a, &[2, 2, 3, 1]);

        test_transposed(a, &b, &[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
    }

    #[test]
    fn complex64_matrix() {
        let a = &[
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.42, 1.5),
            Complex64::new(-2.0, -4.0),
        ];

        let b = transpose(
            &[1, 0],
            Complex64::new(1.0, 0.0),
            a,
            &[3, 2],
            Complex64::new(0.0, 0.0),
            None,
            1,
            true,
        );

        test_transposed(a, &b, &[0, 2, 4, 1, 3, 5]);
    }
}
