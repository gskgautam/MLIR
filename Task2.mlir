module {
  func @matmul(%A: tensor<16384x16384xi32>, %B: tensor<16384x16384xi32>, %C: tensor<16384x16384xi32) -> tensor<16384x16384xi32 {
    %a_shape = tensor.shape %A : tensor<16384x16384xi32
    %b_shape = tensor.shape %B : tensor<16384x16384xi32
    %c_shape = tensor.shape %C : tensor<16384x16384xi32

    %a_dim0 = dim %a_shape, 0 : index
    %a_dim1 = dim %a_shape, 1 : index
    %b_dim0 = dim %b_shape, 0 : index
    %b_dim1 = dim %b_shape, 1 : index
    %c_dim0 = dim %c_shape, 0 : index
    %c_dim1 = dim %c_shape, 1 : index

    %cst0 = constant 0.0 : f32

    %1 = cmpi "sgt", %a_dim0, %c_dim0 : index
    %2 = cmpi "sgt", %a_dim1, %b_dim1 : index
    %3 = cmpi "sgt", %b_dim0, %c_dim1 : index
    %4 = or %1, %2 : index
    %5 = or %4, %3 : index
    %6 = scf.if %5 -> (tensor<16384x16384xi32) {
      %7 = affine.parallel (%i, %j) = (0, 0) to (%c_dim0, %c_dim1) {
        %8 = affine.load %C[%i, %j] : tensor<16384x16384xi32
        %9 = affine.load %A[%i, 0] : tensor<16384x16384xi32
        %10 = affine.load %B[0, %j] : tensor<16384x16384xi32
        %11 = arith.mulf %9, %10 : f32
        %12 = arith.addf %8, %11 : f32
        affine.store %12, %C[%i, %j] : tensor<16384x16384xi32
      }
      %13 = linalg.tensor_reshape %C : tensor<16384x16384xi32 into tensor<16384x16384xi32
      linalg.yield %13 : tensor<16384x16384xi32
    }
    linalg.tensor_reshape %6 : tensor<16384x16384xi32 into tensor<16384x16384xi32
  }
}
