module {
  func @matmul(%A: tensor<16384x16384xf32>, %B: tensor<16384x16384xf32>, %C: tensor<16384x16384xf32>) -> tensor<16384x16384xf32> {
    %a_shape = tensor.shape %A : tensor<16384x16384xf32>
    %b_shape = tensor.shape %B : tensor<16384x16384xf32>
    %c_shape = tensor.shape %C : tensor<16384x16384xf32>

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
    %6 = scf.if %5 -> (tensor<16384x16384xf32>) {
      %7 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"],
        ins(%a: tensor<16384x16384xf32>, %b: tensor<16384x16384xf32>, %c: tensor<16384x16384xf32>) -> tensor<16384x16384xf32> {
          %8 = linalg.tensor_reshape %a : tensor<16384x16384xf32> into tensor<16384x16384xf32>
          %9 = linalg.tensor_reshape %b : tensor<16384x16384xf32> into tensor<16384x16384xf32>
          %10 = linalg.tensor_reshape %c : tensor<16384x16384xf32> into tensor<16384x16384xf32>
          %11 = linalg.init_tensor %10 : tensor<16384x16384xf32>
          %12 = linalg.yield %11 : tensor<16384x16384xf32>
        },
        outs(%c: tensor<16384x16384xf32>) -> tensor<16384x16384xf32> {
          %8 = linalg.tensor_reshape %a : tensor<16384x16384xf32> into tensor<16384x16384xf32>
          %9 = linalg.tensor_reshape %b : tensor<16384x16384xf32> into tensor<16384x16384xf32>
          %10 = linalg.tensor_reshape %c : tensor<16384x16384xf32> into tensor<16384x16384xf32>
          %11 = linalg.init_tensor %10 : tensor<16384x16384xf32>
          %12 = linalg.yield %11 : tensor<16384x16384xf32>
        },
        region {
          ^bb0(%a: f32, %b: f32, %c: f32):
            %13 = arith.mulf %a, %b : f32
            %14 = arith.addf %c, %13 : f32
            linalg.yield %14 : f32
        }
      } -> tensor<16384x16384xf32>
      scf.yield %7 : (tensor<16384x16384xf32>)
    }
    linalg.tensor_reshape %6 : tensor<16384x16384xf32> into tensor<16384x16384xf32>
  }
}
