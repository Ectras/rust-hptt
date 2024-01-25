static mut DEFAULT_NUM_THREADS: u32 = 1;
static mut USE_ROW_MAJOR: bool = false;

mod implementations {
    use std::mem::transmute;

    use num_complex::{Complex32, Complex64};

    use hptt_sys::{
        __BindgenComplex, cTensorTranspose, dTensorTranspose, sTensorTranspose, zTensorTranspose,
    };

    use crate::{DEFAULT_NUM_THREADS, USE_ROW_MAJOR};

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
                outer_size_a.is_none() && outer_size_b.is_none(),
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
                    use_row_major.into(),
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
                outer_size_a.is_none() && outer_size_b.is_none(),
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
                    use_row_major.into(),
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
                outer_size_a.is_none() && outer_size_b.is_none(),
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
                    use_row_major.into(),
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
                outer_size_a.is_none() && outer_size_b.is_none(),
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
                    use_row_major.into(),
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

/// This computes the transpose of `a` multiplied by `alpha` and adds the result to
/// the out tensor `b` multiplied by `beta`. The axes of `a` are permuted in the
/// order given by `perm`. If `b` is `None`, a new vector is created and returned.
///
/// In other words: `b = alpha * transpose(a) + beta * b`
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

/// Computes the transpose of `a`, i.e. returns the data with the axes permuted in
/// the order given by `perm`. It uses the global thread and row/column major
/// settings.
///
/// # Example
/// ```
/// # use hptt::transpose_simple;
/// let a = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // flat data
/// let shape = &[2, 3]; // actual shape of 'a' (2x3 matrix)
/// let perm = &[1, 0]; // swap the axes: put axis 1 first, axis 0 second
/// let b = transpose_simple(perm, a, shape);
/// // 'b' is now the flat data of a (3x2) matrix
/// assert_eq!(b, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
/// ```
pub fn transpose_simple<T>(perm: &[i32], a: &[T], size_a: &[i32]) -> Vec<T>
where
    (): implementations::Transposable<T>,
{
    <() as implementations::Transposable<T>>::transpose_simple(perm, a, size_a)
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

    use float_cmp::assert_approx_eq;
    use num_complex::{Complex32, Complex64};

    use crate::{transpose, transpose_simple};

    fn check_transposed_equality<T>(original: &[T], transposed: &[T], permutated_indices: &[usize])
    where
        T: Debug + PartialEq,
    {
        for (i, &j) in permutated_indices.iter().enumerate() {
            assert_eq!(transposed[i], original[j]);
        }
    }

    #[test]
    fn test_f64_tensor() {
        let a = &[
            0.1, 0.65, 0.34, 0.76, 0.54, 0.17, 0.0, 0.63, 0.37, 0.22, 0.05, 0.17,
        ];

        let b = transpose(&[3, 2, 0, 1], 1.0, a, &[2, 2, 3, 1], 0.0, None, 1, true);

        check_transposed_equality(a, &b, &[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]);
    }

    #[test]
    fn test_simple_f64_tensor() {
        // transpose_simple uses default column-major setting
        let a = &[
            0.1, 0.65, 0.34, 0.76, 0.54, 0.17, 0.0, 0.63, 0.37, 0.22, 0.05, 0.17,
        ];

        let b = transpose_simple(&[3, 2, 0, 1], a, &[2, 2, 3, 1]);

        check_transposed_equality(a, &b, &[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
    }

    #[test]
    fn test_complex64_matrix() {
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

        check_transposed_equality(a, &b, &[0, 2, 4, 1, 3, 5]);
    }

    #[test]
    fn test_multithreaded_f32() {
        let a = [
            2.4f32, 3.5, 4.6, 5.7, 6.8, 7.9, 8.0, 9.1, 10.2, 11.3, 12.4, 13.5,
        ];
        let b = transpose(&[2, 0, 1], 1.0f32, &a, &[2, 3, 2], 0.0f32, None, 4, false);

        check_transposed_equality(&a, &b, &[0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]);
    }

    #[test]
    fn test_alpha_beta_complex32() {
        let a = [
            Complex32::new(1.0, 2.0),
            Complex32::new(0.0, -1.0),
            Complex32::new(0.1, 2.5),
            Complex32::new(0.0, 0.0),
            Complex32::new(-3.0, 0.0),
            Complex32::new(0.0, 3.0),
        ];

        let b = vec![
            Complex32::new(1.0, -0.5),
            Complex32::new(2.0, 0.0),
            Complex32::new(0.5, 1.0),
            Complex32::new(0.0, -2.0),
            Complex32::new(-2.0, 0.0),
            Complex32::new(-0.5, 1.0),
        ];

        let c = transpose(
            &[1, 0],
            Complex32::new(1.0, 0.5),
            &a,
            &[3, 2],
            Complex32::new(0.5, 1.0),
            Some(b),
            1,
            true,
        );

        let solution = [
            Complex32::new(1.0, 3.25),
            Complex32::new(-0.15, 4.55),
            Complex32::new(-3.75, -0.5),
            Complex32::new(2.5, -2.0),
            Complex32::new(-1.0, -2.0),
            Complex32::new(-2.75, 3.0),
        ];

        assert_eq!(c.len(), solution.len());
        for (i, &s) in solution.iter().enumerate() {
            assert_approx_eq!(f32, c[i].re, s.re, ulps = 2);
            assert_approx_eq!(f32, c[i].im, s.im, ulps = 2);
        }
    }
}