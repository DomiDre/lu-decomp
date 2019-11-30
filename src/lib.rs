#![allow(non_snake_case)]

pub mod matrix_solve;
pub use matrix_solve::{matrix_solve, LU_matrix_solve};

pub mod decomposition;
pub use decomposition::LU_decomp;



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn solve_system_of_linear_equations() {
        let A: Array2<f64> = array![[1.0, 3.0, 5.0], [2.0, 4.0, 7.0], [1.0, 1.0, 0.0],];
        let b: Array1<f64> = array![1.0, 2.0, 3.0];
        let x = matrix_solve(&A, &b);
        assert_eq!(A.dot(&x), b);
    }
}