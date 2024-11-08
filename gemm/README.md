# README

# TODO: Only has generic text right now

## 1. Overall Implementations

This project focuses on the implementation of a General Matrix Multiply (GEMM) algorithm. The primary goal is to optimize matrix multiplication operations for performance and efficiency. Various optimization techniques and strategies are employed to achieve this.

## 2. Directory Structure

The directory structure of the project is organized as follows:

- `src/`: Contains the source code files for the GEMM implementation.
- `include/`: Header files and necessary includes.
- `build/`: Directory where the build artifacts will be generated.
- `tests/`: Contains test cases and test scripts to validate the implementation.
- `docs/`: Documentation related to the project.

## 3. Process to Build the Binaries and Run the Workloads

To build the binaries, follow these steps:

1. Navigate to the `build/` directory.
2. Run the build script using the command `./build.sh`.
3. Once the build is complete, execute the binary using `./gemm`.
4. To run the workloads, provide the necessary input files and configurations as arguments to the binary.

## 4. Results Obtained

The results obtained from the GEMM implementation are documented in this section. Performance metrics, benchmarks, and comparisons with other implementations are provided to showcase the efficiency and effectiveness of the optimizations applied.

## 5. To-do

- Look into why adf buffers can't be used as arguments to the kernels. For some reason the output becomes completely incorrect when using adf buffers, while the output becomes correct with the same code when regular input types are used.
