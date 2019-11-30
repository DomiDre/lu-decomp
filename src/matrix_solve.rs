use ndarray::{s, Array1, Array2, NdFloat};
use crate::LU_decomp;

/// Solves linear equation A*x = b by LU decomposition
pub fn matrix_solve<T>(A: &Array2<T>, b: &Array1<T>) -> Array1<T>
where
    T: NdFloat,
{
    // solve Ax = b for x, where A is a square matrix
    let (L, U, P) = LU_decomp(A);
    LU_matrix_solve(&L, &U, &P, &b)
}

/// Solves linear equation A*x = b where the partial pivoted LU decomposition of PA = LU is already given
pub fn LU_matrix_solve<T>(L: &Array2<T>, U: &Array2<T>, P: &Array2<T>, b: &Array1<T>) -> Array1<T>
where
    T: NdFloat,
{
    let matrix_dimension = L.nrows();
    // first solve Ly = Pb
    let pivotized_b = P.dot(b);
    let mut y: Array1<T> = pivotized_b.clone();
    for i in 1..matrix_dimension {
        y[i] = y[i] - L.slice(s![i, 0..i]).dot(&y.slice(s![0..i]));
    }
    // then solve Ux = y
    let mut x: Array1<T> = y.clone();
    x[matrix_dimension - 1] =
        x[matrix_dimension - 1] / U[[matrix_dimension - 1, matrix_dimension - 1]];
    for i in (0..matrix_dimension - 1).rev() {
        x[i] = (x[i]
            - U.slice(s![i, i + 1..matrix_dimension])
                .dot(&x.slice(s![i + 1..matrix_dimension])))
            / U[[i, i]];
    }
    x
}
