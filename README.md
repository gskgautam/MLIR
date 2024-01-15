# MLIR
# Task: 1
Question: Write an MLIR function (using Linalg’s generic operation) for matrix-matrix multiplication (GEMM), i.e., a function that multiplies two 2-dimensional matrices A and matrix B and stores the result in matrix C.  Note: Entry to MLIR is a not straight forward. You can find a vector addition function as an example in this repository : (https://github.com/h4midf/MLIRTaskResources.git).

Solution: Task1.mlir defines a function `matmul` for matrix multiplication of tensors A and B, storing the result in tensor C. It ensures the matrices have valid dimensions using runtime checks. The code utilizes the Linalg dialect's `linalg.generic` operation, parallelizing and reducing dimensions, and specifies the matrix multiplication computation within its region. The conditional execution with `scf.if` ensures correctness based on runtime checks. Overall, the code performs a parallelized matrix multiplication with dynamic dimensions, offering flexibility for various matrix sizes and shapes, and produces a tensor representing the result.

# Task: 2
Question: Lower the MLIR from Task 1 to the MLIR Affine dialect and apply the loop-unroll transformation on it.

Solution: Task2.mlir defines a matrix multiplication function `matmul` operating on integer tensors. It checks the dimensions of matrices A, B, and C, ensuring they satisfy matrix multiplication criteria. The code utilizes affine maps and parallel loops for efficient computation. The `scf.if` statement conditionally executes the matrix multiplication logic based on runtime checks, and the result is reshaped using the Linalg dialect's `linalg.tensor_reshape` operation. Overall, the code represents a parallelized integer matrix multiplication with dynamic dimensions, suitable for various matrix sizes, and outputs an integer tensor representing the result.

# Task: 3
Question: Generate the LLVM IR for the transformed code from Task 2.
