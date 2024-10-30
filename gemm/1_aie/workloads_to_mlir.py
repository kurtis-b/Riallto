import re
from pathlib import Path
import workloads
from workloads import WORKLOADS

# NOTE: For some reason the .seq file generated will cause incorrect functionality if the input and output memref's aren't i32
# If i32 is used instead, the .seq file that's generated will lead to correct functionality of the NPU
DTYPE_SIZE = {'uint8': 1, 'bfloat16': 2, 'i16': 2, 'i32': 4}


def replace_memref(line, shapes, dtype):
    pattern = r'memref<(\d+)xi32>'
    return re.sub(pattern, lambda m: f'memref<{shapes[0]*shapes[1]}xi32>', line)
    # if dtype == 'uint8':
    #     return re.sub(pattern, lambda m: f'memref<{shapes[0]*shapes[1]}xi8>', line)
    # elif dtype == 'bfloat16':
    #     return re.sub(pattern, lambda m: f'memref<{shapes[0]*shapes[1]}xi16>', line) 
    #  # The below two will be used for the outputs, which need higher bit width
    # elif dtype == 'i16':
    #     return re.sub(pattern, lambda m: f'memref<{shapes[0]*shapes[1]}xi16>', line) 
    # elif dtype == 'i32':
    #     return re.sub(pattern, lambda m: f'memref<{shapes[0]*shapes[1]}xi32>', line) 
    # return line


def replace_2d_memref(line, shapes, dtype):
    for i in range(len(shapes)): 
        if f'%itbuffer_{i}' not in line:
            continue
        pattern = r'memref<(\d+)x(\d+)xi32>'
        # if dtype == 'uint8':
        #     line = re.sub(pattern, lambda m: f'memref<{shapes[i][0]}x{shapes[i][1]}xi8>', line, count=1)
        # elif dtype == 'bfloat16':
        #     line = re.sub(pattern, lambda m: f'memref<{shapes[i][0]}x{shapes[i][1]}xi16>', line, count=1)
        # # The below two will be used for the outputs, which need higher bit width
        # elif dtype == 'i16':
        #     line = re.sub(pattern, lambda m: f'memref<{shapes[i][0]}x{shapes[i][1]}xi16>', line, count=1)
        # elif dtype == 'i32': 
        #     line = re.sub(pattern, lambda m: f'memref<{shapes[i][0]}x{shapes[i][1]}xi32>', line, count=1)
        line = re.sub(pattern, lambda m: f'memref<{shapes[i][0]}x{shapes[i][1]}xi32>', line, count=1)
    return line


def replace_func_memref(line, shapes, inp_dtype, out_dtype):
    pattern = r'(func.*?@mmul.*?\(.*?)'
    for _ in range(len(shapes) - 1): 
        pattern += r'memref<(\d+)xi32>, '
    pattern += r'memref<(\d+)xi32>\)'

    def replacement(match):
        prefix = match.group(1)
        sizes = [shape[0]*shape[1] for shape in shapes]
        # memrefs = []
        # if inp_dtype == 'uint8':
        #     memrefs = [f'memref<{size}xi8>' for size in sizes[:len(shapes) - 1]]
        # elif inp_dtype == 'bfloat16':
        #     memrefs = [f'memref<{size}xi16>' for size in sizes[:len(shapes) - 1]]
        # if out_dtype == 'i16':
        #     memrefs = memrefs + [f'memref<{sizes[len(shapes) - 1]}xi16>']
        # elif out_dtype == 'i32':
        #     memrefs = memrefs + [f'memref<{sizes[len(shapes) - 1]}xi32>']
        memrefs = [f'memref<{size}xi32>' for size in sizes]
        return prefix + ', '.join(memrefs) + ')'

    return re.sub(pattern, replacement, line)


def replace_sequence_memref(line, shapes, inp_dtype, out_dtype):
    pattern = r'(func.*?@sequence.*?\(.*?)'
    for i in range(len(shapes) - 1): 
        pattern += rf'%itbuffer_{i} : memref<(\d+)x(\d+)xi32>,'
    pattern += rf'%itbuffer_{len(shapes) - 1} : memref<(\d+)x(\d+)xi32>\)'

    def replacement(match):
        prefix = match.group(1)
        memrefs = [f'memref<{shape[0]}x{shape[1]}xi32>' for shape in shapes]
        return prefix + ','.join([f'%itbuffer_{i} : {memref}' for i, memref in enumerate(memrefs)]) + ')'
    return re.sub(pattern, replacement, line)

    for i in range(len(shapes) - 1): # Don't process the last memref since its dimensions are in the higher bit width
        pattern = rf'(%itbuffer_{i})?memref<(\d+)x(\d+)xi32>'
        print(pattern)
        # if inp_dtype == 'uint8':
        #     line = re.sub(pattern, lambda m: f'memref<{shapes[i][0]}x{shapes[i][1]}xi8>', line, count=1)
        # elif inp_dtype == 'bfloat16':
        #     line = re.sub(pattern, lambda m: f'memref<{shapes[i][0]}x{shapes[i][1]}xi16>', line, count=1)
        line = re.sub(pattern, lambda m: f'memref<{shapes[i][0]}x{shapes[i][1]}xi32>', line, count=1)
        print(line)
    # For the last memref make sure the shape is correct 
    # if out_dtype == 'i16':
    #     line = re.sub(rf'(%itbuffer_{len(shapes) - 1})?memref<(\d+)x(\d+)xi32>', lambda m: f'memref<{shapes[len(shapes) - 1][0]}x{shapes[len(shapes) - 1][1]}xi16>', line, count=1)
    # elif out_dtype == 'i32':
    #     line = re.sub(rf'(%itbuffer_{len(shapes) - 1})?memref<(\d+)x(\d+)xi32>', lambda m: f'memref<{shapes[len(shapes) - 1][0]}x{shapes[len(shapes) - 1][1]}xi32>', line, count=1)
    line = re.sub(rf'(%itbuffer_{len(shapes) - 1})?memref<(\d+)x(\d+)xi32>', lambda m: f'memref<{shapes[len(shapes) - 1][0]}x{shapes[len(shapes) - 1][1]}xi32>', line, count=1)
    print(line)
    return line

def replace_sequence_sizes(line, shapes):
    # The first array is for the offsets. Using that to find the location of the size operands
    pattern = r'\[%c0, %c0, %c0, %c0\]\[(%c\d+), (%c\d+), (%c\d+), (%c\d+)\]'
    new_shape = f'[%c0, %c0, %c0, %c0][%c1, %c1, %c{shapes[0]}, %c{shapes[1]}]'
    line = re.sub(pattern, new_shape, line, count=1)
    return line


def process_mlir_file(input_file, output_file, inp_dtype, out_dtype, npu_shapes, kernel_shapes):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Adjust the input kernel_shapes to 32-bit width based on the inp_dtype
    if inp_dtype != 'i32':
        for i, kernel_shape in enumerate(kernel_shapes[:-1]):
            for j in range(DTYPE_SIZE['i32'] // DTYPE_SIZE[inp_dtype] // 2):
                if j % 2 == 0:
                    kernel_shape = (kernel_shape[0], kernel_shape[1] // 2)
                else: 
                    kernel_shape = (kernel_shape[0] // 2, kernel_shape[1])
            kernel_shapes[i] = kernel_shape
    # Adjust the output kernel shape to 32-bit width if the out_dtype isn't 'i32'
    if out_dtype != 'i32':
        for j in range(DTYPE_SIZE['i32'] // DTYPE_SIZE[out_dtype] // 2):
            if j % 2 == 0:
                kernel_shapes[-1] = (kernel_shapes[-1][0], kernel_shapes[-1][1] // 2)
            else: 
                kernel_shapes[-1] = (kernel_shapes[-1][0] // 2, kernel_shapes[-1][1])
    ssa_consts = [] # Save the SSA constants to be check whether all of the ones needed are in the file
    with open(output_file, 'w') as file:
        line_iter = iter(lines)
        line = next(line_iter, None)
        while line is not None:
            if 'memref' in line:
                if 'sequence' not in line:
                    if 'func.' not in line:
                        if 'pA' in line:
                            line = replace_memref(line, kernel_shapes[0], inp_dtype)
                            if 'subview' in line:
                                file.write(line)
                                line = next(line_iter, None)
                                line = replace_memref(line, kernel_shapes[0], inp_dtype)
                        elif 'pB' in line:
                            line = replace_memref(line, kernel_shapes[1], inp_dtype)
                            if 'subview' in line:
                                file.write(line)
                                line = next(line_iter, None)
                                line = replace_memref(line, kernel_shapes[1], inp_dtype)
                        elif 'pC' in line:
                            line = replace_memref(line, kernel_shapes[2], out_dtype)
                            if 'subview' in line:
                                file.write(line)
                                line = next(line_iter, None)
                                line = replace_memref(line, kernel_shapes[2], out_dtype)
                    else:
                        line = replace_func_memref(line, kernel_shapes, inp_dtype, out_dtype)
                else:
                    break
            file.write(line)
            line = next(line_iter, None)
        line = replace_sequence_memref(line, kernel_shapes, inp_dtype, out_dtype) # In the sequence function since the memrefs will be in two dimensions in the generated MLIR
        file.write(line)
        line = next(line_iter, None)
        while line is not None:
            if 'memref' in line:
                line = replace_2d_memref(line, kernel_shapes, inp_dtype if 'pC' not in line else out_dtype)
                if 'pA' in line:
                    line = replace_sequence_sizes(line, kernel_shapes[0])
                elif 'pB' in line:
                    line = replace_sequence_sizes(line, kernel_shapes[1])
                elif 'pC' in line:
                    line = replace_sequence_sizes(line, kernel_shapes[2])
            if 'arith.constant' in line:
                ssa_consts.append(line[line.index('c')+1:line.index('=')].strip())
            file.write(line)
            line = next(line_iter, None)
    # Check if all the SSA constants needed are in the file
    required_ssa_consts_kernel = {f'{dim}' for shape in kernel_shapes for dim in shape}
    required_ssa_consts_npu = {f'{dim}' for shape in npu_shapes for dim in shape}
    missing_ssa_consts_kernel = required_ssa_consts_kernel - set(ssa_consts)
    missing_ssa_consts_npu = required_ssa_consts_npu - set(ssa_consts) - missing_ssa_consts_kernel
    if missing_ssa_consts_kernel or missing_ssa_consts_npu:
        with open(output_file, 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            sequence_found = False
            for line in lines:
                if line.startswith("func.func @sequence"):
                    sequence_found = True
                if sequence_found and "AIEX.ipu.dma_memcpy_nd" in line:
                    for const in missing_ssa_consts_kernel:
                        file.write(f'    %c{const} = arith.constant {const} : i32\n')
                    for const in missing_ssa_consts_npu:
                        file.write(f'    %c{const} = arith.constant {const} : i32\n')
                    sequence_found = False
                file.write(line)
            

def calculate_workload_size_and_intensity(npu_shapes, kernel_shapes, inp_dtype, out_dtype):
    def size_in_kb(shapes, inp_dtype, out_dtype):
        total_size = sum(shape[0] * shape[1] for shape in shapes[:-1]) * DTYPE_SIZE[inp_dtype]
        total_size = total_size + shapes[-1][0] * shapes[-1][1] * DTYPE_SIZE[out_dtype] # Last memref is in i32 for accumulation
        return total_size / 1000  # Convert to KB

    def arithmetic_intensity(shapes, inp_dtype, out_dtype):
        total_ops = shapes[0][0] * shapes[0][1] * shapes[1][1] * 2 # Multiply by two for MAC operations
        return total_ops / (size_in_kb(shapes, inp_dtype, out_dtype) * 1000) # Calculate in FLOP/Byte

    npu_size_kb = size_in_kb(npu_shapes, inp_dtype, out_dtype)
    kernel_size_kb = size_in_kb(kernel_shapes, inp_dtype, out_dtype)
    npu_intensity = arithmetic_intensity(npu_shapes, inp_dtype, out_dtype)
    kernel_intensity = arithmetic_intensity(kernel_shapes, inp_dtype, out_dtype)

    return npu_size_kb, kernel_size_kb, npu_intensity, kernel_intensity


def calculate_theoretical_kernel_exec_time_us(kernel_shapes, inp_dtype, out_dtype):
    dtype_performance = {'uint8': 256, 'bfloat16': 128} # MACs per cycle
    total_input_size = sum(shape[0] * shape[1] for shape in kernel_shapes[:-1]) * DTYPE_SIZE[inp_dtype]
    total_output_size = kernel_shapes[-1][0] * kernel_shapes[-1][1] * DTYPE_SIZE[out_dtype]
    # Compute IT to CT data theoretical transfer time. Using the aggreggate BW of one IT to CT
    it_to_ct_transfer_time = (total_input_size / 10e9) / 32 # 32 GB/s
    # Compute the theoretical execution time
    theoretical_exec_time = (kernel_shapes[0][0] * kernel_shapes[0][1] * kernel_shapes[1][1]) / (dtype_performance[inp_dtype] * 10e9) # Clock is 1 GHz, only using 1 CT so don't need to divide by number of CTs used
    # Compute CT to IT data theoretical transfer time. Using the aggreggate BW of one CT to IT
    it_to_ct_transfer_time = (total_output_size / 10e9) / 24 # 24 GB/s
    return max(theoretical_exec_time, it_to_ct_transfer_time, it_to_ct_transfer_time) * 10e6 # Convert to microseconds


def calculate_data_tiles(workload_shapes, npu_shapes):
    return (workload_shapes[0][0] // npu_shapes[0][0]) * (workload_shapes[0][1] // npu_shapes[0][1]) * (workload_shapes[1][1] // npu_shapes[1][1])


def calculate_theoretical_workload_exec_time_ms(theoretical_kernel_exec_time_us, workload_shapes, npu_shapes, inp_dtype, out_dtype):
    data_tiles = calculate_data_tiles(workload_shapes, npu_shapes)
    # Assume that the following three executions happen in parallel: A and B inputs transfer to the NPU in sequence, C transfer back to DDR, and kernel execution 
    total_input_size = sum(shape[0] * shape[1] for shape in npu_shapes[:-1]) * DTYPE_SIZE[inp_dtype]
    total_output_size = npu_shapes[-1][0] * npu_shapes[-1][1] * DTYPE_SIZE[out_dtype]
    ddr_to_it_transfer_time = (total_input_size / 10e9) / 89.6 # 89.6 GB/s
    it_to_ddr_transfer_time = (total_output_size / 10e9) / 89.6 # 89.6 GB/s
    return (data_tiles * max((theoretical_kernel_exec_time_us / 10e6) + ddr_to_it_transfer_time, it_to_ddr_transfer_time)) * 10e3 # Convert to milliseconds


def execute():
    # Using the base MLIR file, generate the MLIR file for the workload
    for workload in WORKLOADS:
        input_file = Path(__file__).parent / f'Mmul_1aie.mlir'
        output_file = Path(__file__).parent / workload[workloads.DIRECTORY_KEY] / workload[workloads.FILE_NAME_KEY]
        inp_dtype = workload[workloads.DATA_TYPE_INPUT_KEY]
        out_dtype = workload[workloads.DATA_TYPE_OUTPUT_KEY]
        npu_shapes = workload[workloads.NPU_SHAPES_KEY]
        kernel_shapes = workload[workloads.KERNEL_SHAPES_KEY]
        process_mlir_file(input_file, output_file, inp_dtype, out_dtype, npu_shapes, kernel_shapes)

    # Save the size and intensity of each workload to a data file
    data_file = Path(__file__).parent / 'theoretical_performance.txt'
    with open(data_file, 'w') as df:
        for workload in WORKLOADS:
            workload_shapes = workload[workloads.WORKLOAD_SHAPES_KEY]
            npu_shapes = workload[workloads.NPU_SHAPES_KEY]
            kernel_shapes = workload[workloads.KERNEL_SHAPES_KEY]
            inp_dtype = workload[workloads.DATA_TYPE_INPUT_KEY]
            out_dtype = workload[workloads.DATA_TYPE_OUTPUT_KEY]
            npu_size_kb, kernel_size_kb, npu_intensity, kernel_intensity = calculate_workload_size_and_intensity(npu_shapes, kernel_shapes, inp_dtype, out_dtype)
            df.write(f"Workload: {workload_shapes}\n")
            df.write(f"NPU: {npu_shapes}\n")
            df.write(f"Kernel: {kernel_shapes}\n")
            df.write(f"Data type input: {inp_dtype}\n")
            df.write(f"Data type output: {out_dtype}\n")
            df.write(f"NPU Size: {npu_size_kb:.2f} KB, Kernel Size: {kernel_size_kb:.2f} KB\n")
            df.write(f"NPU Intensity: {npu_intensity:.2f} Op/B, Kernel Intensity: {kernel_intensity:.2f} Op/B\n")
            theoretical_kernel_exec_time_us = calculate_theoretical_kernel_exec_time_us(kernel_shapes, inp_dtype, out_dtype)
            df.write(f"Theoretical Kernel Execution Time: {theoretical_kernel_exec_time_us:.2f} us\n")
            data_tiles = calculate_data_tiles(workload_shapes, npu_shapes)
            df.write(f"Data Tiles: {data_tiles}\n")
            df.write(f"Theoretical NPU Execution Time: {calculate_theoretical_workload_exec_time_ms(theoretical_kernel_exec_time_us, workload_shapes, npu_shapes, inp_dtype, out_dtype):.2f} ms\n")
            df.write("\n")

if __name__ == '__main__':
    execute()
