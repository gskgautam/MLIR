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

Solution: 

# Task: 5
Question: Similar to Task 1, write an MLIR function to perform vector-vector multiplication using your custom operator from Task 4. Lower this code to the LLVM IR.

Solution: 
