# Wrapper crate for [HPTT](https://github.com/springer13/hptt)

This crate provides Rust bindings for the **HPTT library** (High-Performance Tensor Transpose) written in C++. It allows to efficiently transform tensor data to a new shape (similar to e.g. numpy's `transpose()`). Check the original repository for the list of features. It has to be noted that this interface only uses the C interface, which does not expose control over the tuning plan.

## Using this crate

The hptt repository is shipped as submodule together with this crate. When building the crate, the hptt C++ library is built using cmake. **This can lead to longer build times, especially in release mode!** Also note that cmake must already be installed on the system. The crate can then be added as normal dependency to Cargo.toml.

## Interface

The crate exposes two functions: `transpose` and `transpose_simple` with less arguments. These functions are generic over the data type, with the four specializations offered by the C interface of hptt: `f32`, `f64`, `Complex32`, `Complex64`. Check the documentation for explanation of the arguments.

```rust
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

fn transpose_simple(
    perm: &[i32],
    a: &[T],
    size_a: &[i32]
) -> Vec<T>;
```

## Example

The following transposes a 2x3 matrix. The data is passed as a flat slice. Note that `transpose_simple` assumes column-major ordering.

```rust
use hptt_sys::transpose_simple;

// Input:
// 1 3 5
// 2 4 6

let a = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // flat data (column-major)
let shape = &[2, 3]; // actual shape of 'a' (2x3 matrix)
let perm = &[1, 0]; // swap the axes: put axis 1 first, axis 0 second

let b = transpose_simple(perm, a, shape);
// b is now the flat data of the transposed 3x2 matrix

// Output:
// 1 2
// 3 4
// 5 6

assert_eq!(b, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```