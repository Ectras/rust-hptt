use num_traits::{ConstOne, ConstZero};

mod implementations {
    use std::mem::transmute;

    use num_complex::{Complex32, Complex64};

    use hptt_sys::{
        __BindgenComplex, cTensorTranspose, dTensorTranspose, sTensorTranspose, zTensorTranspose,
    };

    pub trait Transposable<T> {
        #[allow(
            clippy::too_many_arguments,
            reason = "This is matching the API of hptt"
        )]
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
    }

    /// Returns a vector with the requested capacity. If vec is given, it should
    /// either have that many elements or be empty, in which case its capacity will
    /// be set accordingly.
    fn with_capacity<T>(vec: Option<Vec<T>>, capacity: usize, zero_initialize: bool) -> Vec<T>
    where
        T: Clone + Default,
    {
        // Get a vector with enough capacity
        if let Some(mut v) = vec {
            if v.len() >= capacity {
                // Given output is fully initialized, can be used directly
                v
            } else if v.is_empty() {
                // Given output is empty, make sure the capacity suffices
                v.reserve_exact(capacity.saturating_sub(v.capacity()));
                v
            } else {
                panic!("Output vector must either have same length as input, or must be empty");
            }
        } else {
            // Create a new vector
            if zero_initialize {
                vec![T::default(); capacity]
            } else {
                Vec::with_capacity(capacity)
            }
        }
    }

    /// Computes the length of the output tensor given the size of the input or
    /// optionally the size of the output, if it is larger.
    fn output_len(size_a: &[i32], outer_size_b: Option<&[i32]>) -> usize {
        if let Some(outer_b) = outer_size_b {
            outer_b
                .iter()
                .map(|&s| TryInto::<usize>::try_into(s).unwrap())
                .product()
        } else {
            size_a
                .iter()
                .map(|&s| TryInto::<usize>::try_into(s).unwrap())
                .product()
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
            let b_total_len = output_len(size_a, outer_size_b);
            let mut out = with_capacity(b, b_total_len, outer_size_b.is_some());
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
                out.set_len(b_total_len);
            }
            out
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
            let b_total_len = output_len(size_a, outer_size_b);
            let mut out = with_capacity(b, b_total_len, outer_size_b.is_some());
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
                out.set_len(b_total_len);
            }
            out
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
            let b_total_len: usize = output_len(size_a, outer_size_b);
            let mut out = with_capacity(b, b_total_len, outer_size_b.is_some());
            unsafe {
                cTensorTranspose(
                    perm.as_ptr(),
                    perm.len().try_into().unwrap(),
                    transmute::<Complex32, __BindgenComplex<f32>>(alpha),
                    false,
                    a.as_ptr().cast::<__BindgenComplex<f32>>(),
                    size_a.as_ptr(),
                    outer_size_a.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    transmute::<Complex32, __BindgenComplex<f32>>(beta),
                    out.as_mut_ptr().cast::<__BindgenComplex<f32>>(),
                    outer_size_b.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    num_threads.try_into().unwrap(),
                    use_row_major.into(),
                );
                out.set_len(b_total_len);
            }
            out
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
            let b_total_len: usize = output_len(size_a, outer_size_b);
            let mut out = with_capacity(b, b_total_len, outer_size_b.is_some());
            unsafe {
                zTensorTranspose(
                    perm.as_ptr(),
                    perm.len().try_into().unwrap(),
                    transmute::<Complex64, __BindgenComplex<f64>>(alpha),
                    false,
                    a.as_ptr().cast::<__BindgenComplex<f64>>(),
                    size_a.as_ptr(),
                    outer_size_a.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    transmute::<Complex64, __BindgenComplex<f64>>(beta),
                    out.as_mut_ptr().cast::<__BindgenComplex<f64>>(),
                    outer_size_b.map_or(std::ptr::null(), <[i32]>::as_ptr),
                    num_threads.try_into().unwrap(),
                    use_row_major.into(),
                );
                out.set_len(b_total_len);
            }
            out
        }
    }
}

/// This computes the transpose of `a` multiplied by `alpha` and adds the result to
/// the out tensor `b` multiplied by `beta`. The axes of `a` are permuted in the
/// order given by `perm`. If `b` is `None`, a new vector is created and returned.
///
/// In other words: `b = alpha * transpose(a) + beta * b`.
///
/// The outer size arguments can be used to operate on sub-tensors:
/// * `outer_size_a` stores the outer-sizes of each dimension of `a`. This parameter
///   may be None, indicating that the outer-size is equal to `size_a`. If it is
///   given, `outer_size_a[i] >= size_a[i]` for all `i` must hold.
/// * `outer_size_b` stores the outer-sizes of each dimension of `b`. This parameter
///   may be None, indicating that the outer-size is equal to `perm(sizeA)`.  If it
///   is given, `outer_size_b[i] >= perm(size_a)[i]` for all `i` must hold.
#[allow(
    clippy::too_many_arguments,
    reason = "This is matching the API of hptt"
)]
pub fn transpose<T>(
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
) -> Vec<T>
where
    (): implementations::Transposable<T>,
{
    // Check perm
    assert_eq!(
        perm.len(),
        size_a.len(),
        "len(perm) must be equal to len(size_a)"
    );

    // Check outer_size_a
    if let Some(outer_a) = outer_size_a {
        assert!(
            outer_a.len() == size_a.len(),
            "len(outer_size_a) must be equal to len(size_a)"
        );
        for (&outer, &size) in outer_a.iter().zip(size_a) {
            assert!(
                outer >= size,
                "outer_size_a[i] must be greater equal size_a[i]"
            );
        }
    }

    // Check outer_size_b
    if let Some(outer_b) = outer_size_b {
        assert!(
            outer_b.len() == size_a.len(),
            "len(outer_size_b) must be equal to len(size_a)"
        );
        for (&outer, size) in outer_b.iter().zip(perm.iter().map(|&i| size_a[i as usize])) {
            assert!(
                outer >= size,
                "outer_size_b[i] must be greater equal size_a[perm[i]]"
            );
        }
    }

    <() as implementations::Transposable<T>>::transpose(
        perm,
        alpha,
        a,
        size_a,
        outer_size_a,
        beta,
        b,
        outer_size_b,
        num_threads,
        use_row_major,
    )
}

/// Computes the transpose of `a`, i.e. returns the data with the axes permuted in
/// the order given by `perm`. It assumes row-major memory layout and uses a single
/// thread.
///
/// # Example
/// ```
/// # use hptt::transpose_simple;
/// let a = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // flat data
/// let shape = &[2, 3]; // actual shape of 'a' (2x3 matrix)
/// let perm = &[1, 0]; // swap the axes: put axis 1 first, axis 0 second
/// let b = transpose_simple(perm, a, shape);
/// // 'b' is now the flat data of a (3x2) matrix
/// assert_eq!(b, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
/// ```
#[inline]
pub fn transpose_simple<T>(perm: &[i32], a: &[T], size_a: &[i32]) -> Vec<T>
where
    (): implementations::Transposable<T>,
    T: ConstZero + ConstOne,
{
    transpose(perm, T::ONE, a, size_a, None, T::ZERO, None, None, 1, true)
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use float_cmp::{assert_approx_eq, ApproxEq};
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

    fn check_approx_equality<T>(a: &[T], b: &[T])
    where
        T: ApproxEq + Debug + Copy,
    {
        assert_eq!(a.len(), b.len());
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            assert_approx_eq!(T, ai, bi);
        }
    }

    #[test]
    fn test_f64_tensor() {
        let a = &[
            0.1, 0.65, 0.34, 0.76, 0.54, 0.17, 0.0, 0.63, 0.37, 0.22, 0.05, 0.17,
        ];

        let b = transpose_simple(&[3, 2, 0, 1], a, &[2, 2, 3, 1]);

        check_transposed_equality(a, &b, &[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]);
    }

    #[test]
    fn test_simple_f64_tensor() {
        // transpose_simple uses default row-major setting
        let a = &[
            0.1, 0.65, 0.34, 0.76, 0.54, 0.17, 0.0, 0.63, 0.37, 0.22, 0.05, 0.17,
        ];

        let b = transpose(
            &[3, 2, 0, 1],
            1.0,
            a,
            &[2, 2, 3, 1],
            None,
            0.0,
            None,
            None,
            1,
            false,
        );

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
            None,
            Complex64::new(0.0, 0.0),
            None,
            None,
            1,
            true,
        );

        check_transposed_equality(a, &b, &[0, 2, 4, 1, 3, 5]);
    }

    #[test]
    fn test_multithreaded_f32() {
        let a = &[
            2.4f32, 3.5, 4.6, 5.7, 6.8, 7.9, 8.0, 9.1, 10.2, 11.3, 12.4, 13.5,
        ];
        let b = transpose(
            &[2, 0, 1],
            1.0f32,
            a,
            &[2, 3, 2],
            None,
            0.0f32,
            None,
            None,
            4,
            false,
        );

        check_transposed_equality(a, &b, &[0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]);
    }

    #[test]
    fn test_outer_size_a_f32() {
        let a = &[
            2.4f32, 3.5, 4.6, 5.7, 6.8, 7.9, 8., 9.1, 10.2, 11.3, 12.4, 13.5,
        ];
        let b = transpose(
            &[2, 0, 1],
            1.0f32,
            a,
            &[2, 2, 1],
            Some(&[2, 3, 2]),
            0.0f32,
            None,
            None,
            1,
            false,
        );

        let solution = vec![2.4f32, 3.5, 4.6, 5.7];

        check_approx_equality(&b, &solution);
    }

    #[test]
    fn test_outer_size_b_f64() {
        let a = &[5.0, -3., 7.5, 0., 6.8, -3.1];
        let b = transpose(
            &[1, 0],
            1.0,
            a,
            &[3, 2],
            None,
            0.0,
            None,
            Some(&[3, 3]),
            1,
            false,
        );

        let solution = vec![5.0, 0., 0., -3., 6.8, 0., 7.5, -3.1, 0.];

        check_approx_equality(&b, &solution);
    }

    #[test]
    fn test_outer_sizes_f64() {
        // Input: a 4x4 matrix
        // Output: twice the transpose of the left lower 2x2 sub-matrix + b, written
        // into the left upper 2x2 sub-matrix of b (3x3)
        #[rustfmt::skip]
        let a = &[
            -1.0,  2.5,  7.5, -3.0,
             0.0,  4.2,  3.7,  1.2,
             4.5,  6.1, -2.3,  0.5,
             1.2,  3.4,  5.6,  7.8,
        ];

        #[rustfmt::skip]
        let b = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ];

        let b = transpose(
            &[1, 0],
            2.0,
            &a[8..], // start from the upper left element of the lower 2x2 sub-matrix
            &[2, 2],
            Some(&[4, 4]),
            1.0,
            Some(b),
            Some(&[3, 3]),
            1,
            true,
        );

        #[rustfmt::skip]
        let solution = [
            10.0,  4.4, 3.0,
            16.2, 11.8, 6.0,
             7.0,  8.0, 9.0
        ];

        assert_eq!(b.len(), solution.len());
    }

    #[test]
    fn test_alpha_beta_complex32() {
        let a = &[
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
            a,
            &[3, 2],
            None,
            Complex32::new(0.5, 1.0),
            Some(b),
            None,
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
