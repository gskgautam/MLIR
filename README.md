# MLIR
# Task: 1
Question: Write an MLIR function (using Linalg’s generic operation) for matrix-matrix multiplication (GEMM), i.e., a function that multiplies two 2-dimensional matrices A and matrix B and stores the result in matrix C.  Note: Entry to MLIR is a not straight forward. You can find a vector addition function as an example in this repository : (https://github.com/h4midf/MLIRTaskResources.git).

Solution: Task1.mlir defines a function `matmul` for matrix multiplication of tensors A and B, storing the result in tensor C. It ensures the matrices have valid dimensions using runtime checks. The code utilizes the Linalg dialect's `linalg.generic` operation, parallelizing and reducing dimensions, and specifies the matrix multiplication computation within its region. The conditional execution with `scf.if` ensures correctness based on runtime checks. Overall, the code performs a parallelized matrix multiplication with dynamic dimensions, offering flexibility for various matrix sizes and shapes, and produces a tensor representing the result.

# Task: 2
Question: Lower the MLIR from Task 1 to the MLIR Affine dialect and apply the loop-unroll transformation on it.

Solution: Task2.mlir defines a matrix multiplication function `matmul` operating on integer tensors. It checks the dimensions of matrices A, B, and C, ensuring they satisfy matrix multiplication criteria. The code utilizes affine maps and parallel loops for efficient computation. The `scf.if` statement conditionally executes the matrix multiplication logic based on runtime checks, and the result is reshaped using the Linalg dialect's `linalg.tensor_reshape` operation. Overall, the code represents a parallelized integer matrix multiplication with dynamic dimensions, suitable for various matrix sizes, and outputs an integer tensor representing the result.

# Task: 3
Question: Generate the LLVM IR for the transformed code from Task 2.

Solution: Task3.mlir convert MLIR code to LLVM IR, the `mlir-translate` tool is employed, leveraging the `-mlir-to-llvmir` option. After ensuring the installation of MLIR and LLVM tools, execute the command: `mlir-translate -mlir-to-llvmir <your_mlir_file>.mlir > <output_ll_file>.ll`. Replace `<your_mlir_file>.mlir` with your specific MLIR file's name and `<output_ll_file>.ll` with the desired LLVM IR output file name. For instance, if the MLIR file is named `task2.mlir`, the command `mlir-translate -mlir-to-llvmir task2.mlir > task2.ll` will generate LLVM IR in the `task2.ll` file. The resulting LLVM IR can be further inspected or compiled using LLVM tools like `llc` or `clang`.

# Task: 4
Question: Introduce a custom operation at the Linalg level in MLIR. For instance, it already has matmul and matVec operations, you can introduce one for vecVec to perform vector-vector multiplication.

Solution: Task4.mlir defines a custom operation `vecVec` for vector-vector multiplication. The operation multiplies corresponding elements of two vectors `%A` and `%B` and stores the result in an output vector `%C`. The code checks that input vectors have the same shape, using an assertion. The `linalg.generic` operation is employed with parallel iterators, iterating over vector elements. It loads elements from input vectors, performs element-wise multiplication, and stores the results in the output vector. The example usage demonstrates the custom operation for vector-vector multiplication with 16384x16384 elements of 32-bit integers. The code is SPDX-licensed under Apache-2.0.

# Task: 5
Question: Similar to Task 1, write an MLIR function to perform vector-vector multiplication using your custom operator from Task 4. Lower this code to the LLVM IR.

Solution: Task5.mlir introduces a function `vectorVecMul` using the custom operator `vecVec` for vector-vector multiplication. It takes two input vectors `%A` and `%B` of 4 elements each, performs element-wise multiplication using the `vecVec` operator, and stores the result in the output vector `%C`. To lower this MLIR code to LLVM IR, the `mlir-translate` tool is recommended. Assuming the MLIR file is named `task5.mlir`, executing `mlir-translate -mlir-to-llvmir task5.mlir > task5.ll` generates the LLVM IR representation in the `task5.ll` file, ready for further inspection or compilation using LLVM tools like `llc` or `clang`.
