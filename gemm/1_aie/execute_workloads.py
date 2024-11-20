from pathlib import Path
from workloads import WORKLOADS, APP_NAME, FILE_NAME_KEY, DIRECTORY_KEY, DATA_TYPE_INPUT_KEY, DATA_TYPE_OUTPUT_KEY, WORKLOAD_SHAPES_KEY, NPU_SHAPES_KEY, KERNEL_SHAPES_KEY, KERNEL_MMUL_CONFIG_KEY
from workloads import INP_DTYPES as inp_dtypes, OUT_DTYPES as out_dtypes
from ml_dtypes import bfloat16
from npu.runtime import AppRunner
import numpy as np
import time
import matplotlib.pyplot as plt

def partition_matrix(matrix, tile_size):
    rows, cols = matrix.shape
    tile_rows, tile_cols = tile_size
    tiled_matrix = (matrix.reshape(rows // tile_rows, tile_rows, cols // tile_cols, tile_cols).
            swapaxes(1, 2))
    # Flatten the tiled matrix back to the original shape
    return tiled_matrix.reshape(rows, cols)


def execute():
    performance_file = Path(__file__).parent / "actual_performance.csv"
    with open(performance_file, "w") as f:
        num_runs = 10
        executed_cpu_workloads = {}
        f.write("# AIEs,Input Data Type, Output Data Type,Application Workload,NPU Workload,Kernel Workload,NPU Execution (ms),NPU Execution + DDR Transfer (ms)\n")
        for workload in WORKLOADS:
            workload_dir = Path(__file__).parent / workload[DIRECTORY_KEY]
            new_xclbin_name = f"{APP_NAME}_{workload[FILE_NAME_KEY].split('.mlir')[0]}.xclbin"
            new_seq_name = f"{APP_NAME}_{workload[FILE_NAME_KEY].split('.mlir')[0]}.seq"
            new_xclbin = workload_dir / new_xclbin_name
            new_seq = workload_dir / new_seq_name
            app = AppRunner(str(new_xclbin), fw_sequence=str(new_seq))

            # Allocate app input and output buffers to exchange data with NPU
            a_shape_npu = workload[NPU_SHAPES_KEY][0]
            b_shape_npu = workload[NPU_SHAPES_KEY][1]
            c_shape_npu = workload[NPU_SHAPES_KEY][2]
            inp_dtype = inp_dtypes[workload[DATA_TYPE_INPUT_KEY]]
            out_dtype = out_dtypes[workload[DATA_TYPE_OUTPUT_KEY]]
            inp_dtype_host = inp_dtypes[workload[DATA_TYPE_INPUT_KEY]]
            out_dtype_host = out_dtypes[workload[DATA_TYPE_OUTPUT_KEY]]
            # NOTE: If I use np.float16, as the dtype for the input buffers, the NPU output will be all 0's
            if inp_dtype_host == bfloat16:
                inp_dtype_host = np.float16
            if out_dtype_host == bfloat16:
                out_dtype_host = np.float16
            input_a = app.allocate(shape=a_shape_npu, dtype=inp_dtype)
            input_b = app.allocate(shape=b_shape_npu, dtype=inp_dtype)
            output_c = app.allocate(shape=c_shape_npu, dtype=out_dtype)

            # Load array into input buffer
            a_shape_workload = workload[WORKLOAD_SHAPES_KEY][0]
            b_shape_workload = workload[WORKLOAD_SHAPES_KEY][1]
            c_shape_workload = workload[WORKLOAD_SHAPES_KEY][2]
            b = np.ones(shape=b_shape_workload, dtype=inp_dtype)
            a = np.ones(shape=a_shape_workload, dtype=inp_dtype)
            c = np.zeros(shape=c_shape_workload, dtype=out_dtype)
            # if 'i' in workload[DATA_TYPE_INPUT_KEY] or 'u' in workload[DATA_TYPE_INPUT_KEY]:
            #     max_value = np.iinfo(a.dtype).max
            #     a.fill(max_value)
            #     max_value = np.iinfo(b.dtype).max
            #     b.fill(max_value)
            # elif 'f' in workload[DATA_TYPE_INPUT_KEY]:
            #     max_value = np.finfo(np.float16).max
            #     a.fill(max_value)
            #     max_value = np.finfo(np.float16).max
            #     b.fill(max_value)
            # else:
            #     raise ValueError(f"Unknown data type: {workload[DATA_TYPE_INPUT_KEY]}")
            total_npu_time = 0.0
            total_npu_time_plus_ddr_transfer = 0.0

            M = workload[KERNEL_MMUL_CONFIG_KEY][0]
            K = workload[KERNEL_MMUL_CONFIG_KEY][1]
            N = workload[KERNEL_MMUL_CONFIG_KEY][2]
            for run in range(num_runs):
                c = np.zeros(shape=c_shape_workload, dtype=out_dtype) # Reset output matrix c to zeros
                c_tiled = np.zeros(shape=c_shape_npu, dtype=out_dtype)
                for row_a in range(0, a_shape_workload[0], a_shape_npu[0]):
                    for col_b in range(0, b_shape_workload[1], b_shape_npu[1]):
                        for col_a in range(0, a_shape_workload[1], a_shape_npu[1]):
                            a_tiled = partition_matrix(a[row_a:row_a+a_shape_npu[0],col_a:col_a+a_shape_npu[1]], (M,K))
                            b_tiled = partition_matrix(b[col_a:col_a+a_shape_npu[1],col_b:col_b+b_shape_npu[1]], (K,N))
                            # a_tiled = a[row_a:row_a+a_shape_npu[0],col_a:col_a+a_shape_npu[1]]
                            # b_tiled = b[col_a:col_a+a_shape_npu[1],col_b:col_b+b_shape_npu[1]]
                            start = time.time()
                            input_a[:] = a_tiled
                            input_b[:] = b_tiled
                            # Pass input_image buffer to NPU
                            input_a.sync_to_npu()
                            input_b.sync_to_npu()
                            total_npu_time_plus_ddr_transfer = total_npu_time_plus_ddr_transfer + (time.time() - start)
                            # app._refresh_sequence()
                
                            # Run app on NPU
                            start = time.time()
                            app.call(input_a, input_b, output_c)
                            diff = time.time() - start
                            total_npu_time = total_npu_time + diff
                            total_npu_time_plus_ddr_transfer = total_npu_time_plus_ddr_transfer + diff
                
                            # Get results from NPU via output_image buffer
                            start = time.time()
                            output_c.sync_from_npu()
                            total_npu_time_plus_ddr_transfer = total_npu_time_plus_ddr_transfer + (time.time() - start)
                            c_tiled = output_c[:]
                            c_tiled = c_tiled.reshape(a_shape_npu[0] // M, M, b_shape_npu[1] // N, N).swapaxes(1, 2).reshape(a_shape_npu[0], b_shape_npu[1])
                            c[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]] = c[row_a:row_a+a_shape_npu[0],col_b:col_b+b_shape_npu[1]] + c_tiled
                print(f"Finished NPU run {run}")
            total_npu_time = (total_npu_time / num_runs) * 1e3
            total_npu_time_plus_ddr_transfer = (total_npu_time_plus_ddr_transfer / num_runs) * 1e3

            # Otain the CPU calculation time
            expected_output = np.zeros(shape=c_shape_workload, dtype=out_dtype_host) # Reset expected output matrix to zeros
            total_cpu_time = 0.0
            b = np.ones(shape=b_shape_workload, dtype=inp_dtype_host)
            a = np.ones(shape=a_shape_workload, dtype=inp_dtype_host)
            # if 'i' in workload[DATA_TYPE_INPUT_KEY] or 'u' in workload[DATA_TYPE_INPUT_KEY]:
            #     max_value = np.iinfo(a.dtype).max
            #     a.fill(max_value)
            #     max_value = np.iinfo(b.dtype).max
            #     b.fill(max_value)
            # elif 'f' in workload[DATA_TYPE_INPUT_KEY]:
            #     max_value = np.finfo(np.float16).max
            #     a.fill(max_value)
            #     max_value = np.finfo(np.float16).max
            #     b.fill(max_value)
            # else:
            #     raise ValueError(f"Unknown data type: {workload[DATA_TYPE_INPUT_KEY]}")
            if (f"{c_shape_workload[0]}x{a.shape[1]}x{c_shape_workload[1]}", inp_dtype_host, out_dtype_host) not in executed_cpu_workloads.keys():
                for run in range(num_runs):
                    start = time.time()
                    expected_output = np.matmul(a, b, dtype=out_dtype_host)
                    total_cpu_time = total_cpu_time + (time.time() - start)
                    print(f"Finished CPU run {run}")
                total_cpu_time = (total_cpu_time / num_runs) * 1e3
                executed_cpu_workloads[(f"{c_shape_workload[0]}x{a.shape[1]}x{c_shape_workload[1]}", inp_dtype_host, out_dtype_host)] = (total_cpu_time, expected_output)
            else:
                (total_cpu_time, expected_output) = executed_cpu_workloads[(f"{c_shape_workload[0]}x{a.shape[1]}x{c_shape_workload[1]}", inp_dtype_host, out_dtype_host)]

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

            print(f'Checking for error in kernel calculation: Min(Actual-Expected)={np.min(c-expected_output)}, Max(Actual-Expected)={np.max(c-expected_output)}')
            if np.min(c-expected_output) > 0 and np.max(c-expected_output) > 0:
                print('Kernel calculation is incorrect')
                break
            print('Application performance:')
            print(f'total NPU calculation time (ms)={total_npu_time}')
            print(f'Total NPU calculation time+DDR transfer overhead (ms)={total_npu_time_plus_ddr_transfer}')
            print(f'total CPU calculation time (ms)={total_cpu_time}')

            # Write the application performance to a file
            f.write(f"{1},{inp_dtype},{out_dtype},{workload[WORKLOAD_SHAPES_KEY][0][0]}x{workload[WORKLOAD_SHAPES_KEY][0][1]}x{workload[WORKLOAD_SHAPES_KEY][1][1]},{workload[NPU_SHAPES_KEY][0][0]}x{workload[NPU_SHAPES_KEY][0][1]}x{workload[NPU_SHAPES_KEY][1][1]},{workload[KERNEL_SHAPES_KEY][0][0]}x{workload[KERNEL_SHAPES_KEY][0][1]}x{workload[KERNEL_SHAPES_KEY][1][1]},{total_npu_time},{total_npu_time_plus_ddr_transfer}\n")
        f.write(f"Input Data Type, Output Data Type,Application Workload,CPU Execution Time (ms)\n")
        for (workload, inp_dtype, out_dtype_host), (cpu_time, _) in executed_cpu_workloads.items():
            f.write(f"{inp_dtype},{out_dtype_host},{workload},{cpu_time}\n")
        del app


if __name__ == "__main__":
    execute()
