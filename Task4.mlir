# SPDX-License-Identifier: Apache-2.0
module {
  // Custom operation for vector-vector multiplication
  // The operation multiplies corresponding elements of two vectors and stores the result in an output vector.
  // Syntax: %Result = linalg.vecVec %A, %B : tensor<16384x16384xi32>, tensor<16384x16384xi32> into tensor<16384x16384xi32>
  // Example: %C = linalg.vecVec %A, %B : tensor<4xf32>, tensor<4xf32> into tensor<4xf32>
  func @vecVec(%A: tensor<16384x16384xi32>, %B: tensor<16384x16384xi32>) -> tensor<16384x16384xi32> {
    %a_shape = tensor.shape %A : tensor<16384x16384xi32>
    %b_shape = tensor.shape %B : tensor<16384x16384xi32>

    // Check that the input vectors have the same shape
    %1 = cmpi "eq", %a_shape, %b_shape : tensor<16384x16384xi32>
    scf.assert %1 : tensor<16384x16384xi32> : "Input vectors must have the same shape"

    %C = linalg.generic {
      indexing_maps = [
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>
      ],
      iterator_types = ["parallel"],
      ins(%a: tensor<16384x16384xi32>, %b: tensor<16384x16384xi32>) -> tensor<16384x16384xi32> {
        %element_a = linalg.indexed_generic_load %a[%i] : tensor<16384x16384xi32>
        %element_b = linalg.indexed_generic_load %b[%i] : tensor<16384x16384xi32>
        %result = arith.mulf %element_a, %element_b : f32
        linalg.yield %result : f32
      },
      outs(%c: tensor<16384x16384xi32>) -> tensor<16384x16384xi32> {
        linalg.indexed_generic_store %C[%i], %c : tensor<16384x16384xi32>
      }
    } -> tensor<16384x16384xi32>
    return %C : tensor<16384x16384xi32>
  }
}
