mod hptt {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

static mut DEFAULT_NUM_THREADS: u32 = 1;
static mut USE_ROW_MAJOR: bool = true;

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
            outer_size_b: Option<&[i32]>,
            num_threads: u32,
            use_row_major: bool,
        ) -> Vec<T>;

        fn transpose_simple(perm: &[i32], a: &[T], size_a: &[i32]) -> Vec<T>;
    }

    impl Transposable<f32> for () {
        fn transpose(
            perm: &[i32],
            alpha: f32,
            a: &[f32],
            size_a: &[i32],
            outer_size_a: Option<&[i32]>,
            beta: f32,
            outer_size_b: Option<&[i32]>,
            num_threads: u32,
            use_row_major: bool,
        ) -> Vec<f32> {
            assert!(
                matches!(outer_size_a, None) && matches!(outer_size_b, None),
                "Outer size not supported yet"
            );
            let mut out = Vec::with_capacity(a.len());
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
            outer_size_b: Option<&[i32]>,
            num_threads: u32,
            use_row_major: bool,
        ) -> Vec<f64> {
            assert!(
                matches!(outer_size_a, None) && matches!(outer_size_b, None),
                "Outer size not supported yet"
            );
            let mut out = Vec::with_capacity(a.len());
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
            outer_size_b: Option<&[i32]>,
            num_threads: u32,
            use_row_major: bool,
        ) -> Vec<Complex32> {
            assert!(
                matches!(outer_size_a, None) && matches!(outer_size_b, None),
                "Outer size not supported yet"
            );
            let mut out: Vec<Complex32> = Vec::with_capacity(a.len());
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
            outer_size_b: Option<&[i32]>,
            num_threads: u32,
            use_row_major: bool,
        ) -> Vec<Complex64> {
            assert!(
                matches!(outer_size_a, None) && matches!(outer_size_b, None),
                "Outer size not supported yet"
            );
            let mut out: Vec<Complex64> = Vec::with_capacity(a.len());
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
                unsafe { DEFAULT_NUM_THREADS },
                unsafe { USE_ROW_MAJOR },
            )
        }
    }
}

pub fn transpose<T>(
    perm: &[i32],
    alpha: T,
    a: &[T],
    size_a: &[i32],
    beta: T,
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
        None,
        num_threads,
        use_row_major,
    )
}

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

        let b = transpose(&[3, 2, 0, 1], 1.0, a, &[2, 2, 3, 1], 0.0, 1, true);

        test_transposed(a, &b, &[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]);
    }

    #[test]
    fn f64_tensor_simple() {
        let a = &[
            0.1, 0.65, 0.34, 0.76, 0.54, 0.17, 0.0, 0.63, 0.37, 0.22, 0.05, 0.17,
        ];

        let b = transpose_simple(&[3, 2, 0, 1], a, &[2, 2, 3, 1]);

        test_transposed(a, &b, &[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]);
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
            1,
            true,
        );

        test_transposed(a, &b, &[0, 2, 4, 1, 3, 5]);
    }
}
