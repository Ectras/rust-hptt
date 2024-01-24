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

This example transposes a vector of shape (2, 2, 3, 1) to a vector of shape (1, 3, 2, 2). The data is passed as a flat slice. Note that `transpose_simple` assumes column-major ordering.

```rust
use hptt_sys::transpose_simple;

let a = &[0.1, 0.65, 0.34, 0.76, 0.54, 0.17, 0.0, 0.63, 0.37, 0.22, 0.05, 0.17,];
let b = transpose_simple(&[3, 2, 0, 1], a, &[2, 2, 3, 1]);
assert_eq!(b, vec![0.1, 0.54, 0.37, 0.65, 0.17, 0.22, 0.34, 0, 0.05, 0.76, 0.63, 0.17]);
```