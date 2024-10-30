from pathlib import Path
import workloads
from workloads import WORKLOADS
from generate_base_mlir import Mmul_1aie
from ml_dtypes import bfloat16
from npu.runtime import AppRunner
import numpy as np
import time
import matplotlib.pyplot as plt
from npu import nputop

def partition_matrix(matrix, tile_size):
    rows, cols = matrix.shape
    tile_rows, tile_cols = tile_size
    tiled_matrix = (matrix.reshape(rows // tile_rows, tile_rows, cols // tile_cols, tile_cols).
            swapaxes(1, 2))
    # Flatten the tiled matrix back to the original shape
    return tiled_matrix.reshape(rows, cols)
    

if __name__ == "__main__":
    performance_file = Path(__file__).parent / "actual_performance.txt"
    with open(performance_file, "w") as f:
        for workload in WORKLOADS:
            num_runs = 1
            workload_dir = Path(__file__).parent / workload[workloads.DIRECTORY_KEY]
            new_xclbin_name = f"{Mmul_1aie.__name__}_{workload[workloads.FILE_NAME_KEY].split('.mlir')[0]}.xclbin"
            new_seq_name = f"{Mmul_1aie.__name__}_{workload[workloads.FILE_NAME_KEY].split('.mlir')[0]}.seq"
            new_xclbin = workload_dir / new_xclbin_name
            new_seq = workload_dir / new_seq_name
            app = AppRunner(str(new_xclbin), fw_sequence=str(new_seq))

            # Allocate app input and output buffers to exchange data with NPU
            a_shape_npu = workload[workloads.NPU_SHAPES_KEY][0]
            b_shape_npu = workload[workloads.NPU_SHAPES_KEY][1]
            c_shape_npu = workload[workloads.NPU_SHAPES_KEY][2]
            inp_dtype = np.uint8 if workload[workloads.DATA_TYPE_INPUT_KEY] == 'uint8' else bfloat16
            out_dtype = np.uint16 if workload[workloads.DATA_TYPE_OUTPUT_KEY] == 'i16' else np.float32
            # TODO: If I use np.float16, as the dtype for the input buffers, the NPU output will be all 0's
            input_a = app.allocate(shape=a_shape_npu, dtype=inp_dtype)
            input_b = app.allocate(shape=b_shape_npu, dtype=inp_dtype)
            output_c = app.allocate(shape=c_shape_npu, dtype=out_dtype)
            # print(input_a.shape, input_b.shape, output_c.shape)
            # print(input_a.dtype, input_b.dtype, output_c.dtype)

            # Load array into input buffer
            a_shape_workload = workload[workloads.WORKLOAD_SHAPES_KEY][0]
            b_shape_workload = workload[workloads.WORKLOAD_SHAPES_KEY][1]
            c_shape_workload = workload[workloads.WORKLOAD_SHAPES_KEY][2]
            # b = np.zeros(shape=b_shape_workload, dtype=inp_dtype)
            # for i in range(min(b_shape_workload[0], b_shape_workload[1])):
            #     b[i][i] = i % 255
            inp_dtype_host = np.uint8 if workload[workloads.DATA_TYPE_INPUT_KEY] == 'uint8' else np.float16
            b = np.ones(shape=b_shape_workload, dtype=inp_dtype_host)
            a = np.ones(shape=a_shape_workload, dtype=inp_dtype_host)
            c = np.zeros(shape=c_shape_workload, dtype=out_dtype)
            total_time = 0
            # print(a.shape, b.shape, c.shape)
            # print(a.dtype, b.dtype, c.dtype)

            M = workload[workloads.KERNEL_MMUL_CONFIG_KEY][0]
            K = workload[workloads.KERNEL_MMUL_CONFIG_KEY][1]
            N = workload[workloads.KERNEL_MMUL_CONFIG_KEY][2]
            # test = 4
            for _ in range(num_runs):
                c = np.zeros(shape=c_shape_workload, dtype=out_dtype) # Reset output matrix c to zeros
                expected_output = np.zeros(shape=c_shape_workload, dtype=out_dtype) # Reset expected output matrix to zeros
                for row_a in range(0, a_shape_workload[0], a_shape_npu[0]):
                    for col_b in range(0, b_shape_workload[1], b_shape_npu[1]):
                        for col_a in range(0, a_shape_workload[1], a_shape_npu[1]):
                            a_tiled = partition_matrix(a[row_a:row_a+a_shape_npu[0],col_a:col_a+a_shape_npu[1]], (M,K))
                            b_tiled = partition_matrix(b[col_a:col_a+a_shape_npu[1],col_b:col_b+b_shape_npu[1]], (K,N))
                            # a_tiled = a[row_a:row_a+a_shape_npu[0],col_a:col_a+a_shape_npu[1]]
                            # b_tiled = b[col_a:col_a+a_shape_npu[1],col_b:col_b+b_shape_npu[1]]
                            input_a[:] = a_tiled
                            input_b[:] = b_tiled
                            # Pass input_image buffer to NPU
                            input_a.sync_to_npu()
                            input_b.sync_to_npu()
                            app._refresh_sequence()
                
                            # Run app on NPU
                            start = time.time()
                            app.call(input_a, input_b, output_c)
                            total_time = total_time + (time.time() - start)
                
                            # Get results from NPU via output_image buffer
                            output_c.sync_from_npu()
                            c[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]] = c[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]] + output_c
                            expected_output[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]] = expected_output[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]] + partition_matrix(a[row_a:row_a+a_shape_npu[0],col_a:col_a+a_shape_npu[1]]@b[col_a:col_a+a_shape_npu[1],col_b:col_b+b_shape_npu[1]], (M,N))
                            # expected_output[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]] = expected_output[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]] + a[row_a:row_a+a_shape_npu[0],col_a:col_a+a_shape_npu[1]]@b[col_a:col_a+a_shape_npu[1],col_b:col_b+b_shape_npu[1]]
                            # with np.printoptions(threshold=np.inf):
                            # if test == 4 and np.mean(output_c) != 32:
                            #     print(a_tiled, b_tiled)
                            #     print(np.mean(a_tiled), a_tiled.shape, np.mean(b_tiled), b_tiled.shape)
                            #     print(input_a, input_b)
                            #     print(np.mean(input_a), np.mean(input_b), np.mean(output_c))
                            #     print(c[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]])
                            #     print(expected_output[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]])
                            #     test = test - 1
                            # elif test > 0:
                            #     print(a_tiled, b_tiled)
                            #     print(np.mean(a_tiled), a_tiled.shape, np.mean(b_tiled), b_tiled.shape)
                            #     print(input_a, input_b)
                            #     print(np.mean(input_a), np.mean(input_b), np.mean(output_c))
                            #     print(c[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]])
                            #     print(expected_output[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]])
                            #     test = test - 1
                total_time = total_time / num_runs

                # Otain the CPU calculation time
                start = time.time()
                for _ in range(num_runs):
                    test=a@b
                total_cpu_time = (time.time() - start) / num_runs

                c = c.astype(int)
                expected_output = expected_output.astype(int)

                # # Create a figure with two subplots
                # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

                # # Display the images. Tile the axb since the output from
                # # the kernel will be tiled.
                # ax1.imshow(expected_output, cmap='gray')
                # ax1.set_title('Expected Output')

                # ax2.imshow(c, cmap='gray')
                # ax2.set_title('Output from Kernel')

                # ax3.imshow(c-expected_output, cmap='gray')
                # ax3.set_title('Actual-Expected')

                # # Show the plot
                # plt.show()

                print('Application performance:')
                print(f'Checking for error in kernel calculation: Min(Actual-Expected)={np.min(c-expected_output)}, Max(Actual-Expected)={np.max(c-expected_output)}')
                print(f'total kernel calculation time (without tiling overhead)={total_time}')
                print(f'total cpu calculation time={total_cpu_time}')

                # Write the application performance to a file
                f.write(f"Workload: {workload[workloads.WORKLOAD_SHAPES_KEY]}\n")
                f.write(f"NPU workload={workload[workloads.NPU_SHAPES_KEY]}\n")
                f.write(f"Kernel workload={workload[workloads.KERNEL_SHAPES_KEY]}\n")
                f.write(f"Kernel configuration={workload[workloads.KERNEL_MMUL_CONFIG_KEY]}\n")
                f.write(f"Total kernel calculation time (without tiling overhead)={total_time}\n")
                f.write(f"Total CPU calculation time={total_cpu_time}\n")
                f.write("\n")
                
                del app