lu-decomp
=========
 [![Build Status](https://travis-ci.com/DomiDre/lu-decomp.svg?branch=master)](https://travis-ci.com/DomiDre/lu-decomp)
 [![](http://meritbadge.herokuapp.com/lu-decomp)](https://crates.io/crates/lu-decomp)
 
 The ``lu-decomp`` crate is a small library to calculate the LU decomposition of a ![equation](https://latex.codecogs.com/svg.latex?n&space;\times&space;n) matrix.

 For the representation of a matrix the ndarray crate is used.
 An example to use this crate to solve the equation ![equation](https://latex.codecogs.com/svg.latex?Ax&space;=&space;b) for ![equation](https://latex.codecogs.com/svg.latex?x) would be
 
 ```
  use lu_decomp::matrix_solve;
  use ndarray::{array, Array1, Array2};
  
  let A: Array2<f64> = array![
      [1.0, 3.0, 5.0],
      [2.0, 4.0, 7.0],
      [1.0, 1.0, 0.0]
  ];
  let b: Array1<f64> = array![1.0, 2.0, 3.0];
  let x = matrix_solve(&A, &b);
 ```
 