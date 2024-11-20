from workloads import WORKLOADS
from workloads import EDITED_MLIR_FILE_NAME, DIRECTORY_KEY, FILE_NAME_KEY, NPU_SHAPES_KEY, KERNEL_SHAPES_KEY, DATA_TYPE_INPUT_KEY, DATA_TYPE_OUTPUT_KEY, WORKLOAD_SHAPES_KEY
from ml_dtypes import bfloat16
from pathlib import Path
import csv
import re

# NOTE: For some reason the .seq file generated will cause incorrect functionality if the input and output memref's aren't i32
# If i32 is used instead, the .seq file that's generated will lead to correct functionality of the NPU
# It might've also been that the sequence that was generated was incorrect, but I need to investigate further
DTYPE_SIZE = {
    'u8': 1,
    'i8': 1,
    'u16': 2,
    'i16': 2,
    'u32': 4,
    'i32': 4,
    'f32': 4,
    'f64': 8,
    'bfloat16': 2,
}

DTYPE_PERFORMANCE = { # MACS per cycle
    'u8': 256,
    'i8': 256,
    'u16': 64,
    'i16': 64,
    'bfloat16': 128,
}


def replace_memref(line, shapes):
    pattern = r'memref<(\d+)xi32>'
    return re.sub(pattern, lambda m: f'memref<{shapes[0]*shapes[1]}xi32>', line)


def replace_2d_memref(line, shapes):
    for i in range(len(shapes)): 
        if f'%itbuffer_{i}' not in line:
            continue
        pattern = r'memref<(\d+)x(\d+)xi32>'
        line = re.sub(pattern, lambda m: f'memref<{shapes[i][0]}x{shapes[i][1]}xi32>', line, count=1)
    return line


def replace_func_memref(line, shapes):
    pattern = r'(func.*?@mmul.*?\(.*?)'
    for _ in range(len(shapes) - 1): 
        pattern += r'memref<(\d+)xi32>, '
    pattern += r'memref<(\d+)xi32>'
    if '_start' not in line: # The mmul kernels other than start have an additional memref for the accumulation
        pattern += r', memref<(\d+)xi32>\)'
    else:
        pattern += r'\)'

    def replacement(match):
        prefix = match.group(1)
        sizes = [shape[0]*shape[1] for shape in shapes]
        memrefs = [f'memref<{size}xi32>' for size in sizes]
        if '_start' not in line: # The mmul kernels other than start have an additional memref for the accumulation
            memrefs.append(f'memref<{sizes[-1]}xi32>')
        return prefix + ', '.join(memrefs) + ')'
    return re.sub(pattern, replacement, line)


def replace_sequence_memref(line, shapes):
    pattern = r'(func.*?@sequence.*?\(.*?)'
    for i in range(len(shapes) - 1):
        pattern += r'%itbuffer_\d+ : memref<(\d+)x(\d+)xi32>,'
    pattern += r'%itbuffer_\d+ : memref<(\d+)x(\d+)xi32>\)'

    def replacement(match):
        prefix = match.group(1)
        memrefs = [f'memref<{shape[0]}x{shape[1]}xi32>' for shape in shapes]
        return prefix + ','.join([f'%itbuffer_{i} : {memref}' for i, memref in enumerate(memrefs)]) + ')'
    return re.sub(pattern, replacement, line)

def replace_sequence_sizes(line, shapes, offset):
    # The first array is for the offsets. Using that to find the location of the size operands
    pattern = r'\[%c0, %c0, %c0, (%c\d+)\]\[(%c\d+), (%c\d+), (%c\d+), (%c\d+)\]\[%c0, %c0, (%c\d+)\]'
    new_shape = rf'[%c0, %c0, %c0, %c{shapes[1]//4*offset}][%c1, %c1, %c{shapes[0]}, %c{shapes[1]//4}][%c0, %c0, %c{shapes[1]}]'
    line = re.sub(pattern, new_shape, line, count=1)
    return line


def check_buffer_indexes(line):
    itbuf_idx = int(line[line.find('itbuffer_')+len('itbuffer_')])
    mtbuf_idx = int(line[line.find('mtbuffer_')+len('mtbuffer_')])
    return (itbuf_idx == mtbuf_idx, itbuf_idx)


def process_mlir_file(input_file, output_file, inp_dtype, out_dtype, npu_shapes, kernel_shapes):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Adjust the input kernel_shapes to 32-bit width if the inp_dtype isn't 4 bytes 
    if DTYPE_SIZE[inp_dtype] != 4:
        for i, kernel_shape in enumerate(kernel_shapes[:-1]):
            for j in range(DTYPE_SIZE['i32'] // DTYPE_SIZE[inp_dtype] // 2):
                kernel_shape = (kernel_shape[0], kernel_shape[1] // 2)
            kernel_shapes[i] = kernel_shape
    # Adjust the output kernel shape to 32-bit width if the out_dtype isn't 4 bytes
    if DTYPE_SIZE[out_dtype] != 4:
        for j in range(DTYPE_SIZE['i32'] // DTYPE_SIZE[out_dtype] // 2):
            kernel_shapes[-1] = (kernel_shapes[-1][0], kernel_shapes[-1][1] // 2)
    # Adjust the input npu shapes to 32-bit width if the inp_dtype isn't 4 bytes
    if DTYPE_SIZE[inp_dtype] != 4:
        for i, npu_shape in enumerate(npu_shapes[:-1]):
            for j in range(DTYPE_SIZE['i32'] // DTYPE_SIZE[inp_dtype] // 2):
                npu_shape = (npu_shape[0], npu_shape[1] // 2)
            npu_shapes[i] = npu_shape
    # Adjust the output npu shape to 32-bit width if the out_dtype isn't 4 bytes
    if DTYPE_SIZE[out_dtype] != 4:
        for j in range(DTYPE_SIZE['i32'] // DTYPE_SIZE[out_dtype] // 2):
            npu_shapes[-1] = (npu_shapes[-1][0], npu_shapes[-1][1] // 2)
    ssa_consts = [] # Save the SSA constants to be check whether all of the ones needed are in the file
    with open(output_file, 'w') as file:
        line_iter = iter(lines)
        line = next(line_iter, None)
        while line is not None:
            if 'memref' in line:
                if 'sequence' not in line:
                    if 'func.' not in line:
                        if 'pAccum' in line: # Do before 'pA' since 'pA' is in 'pAccum'
                            line = replace_memref(line, kernel_shapes[2])
                            if 'subview' in line:
                                file.write(line)
                                line = next(line_iter, None)
                                line = replace_memref(line, kernel_shapes[2])
                        elif 'pA' in line:
                            line = replace_memref(line, kernel_shapes[0])
                            if 'subview' in line:
                                file.write(line)
                                line = next(line_iter, None)
                                line = replace_memref(line, kernel_shapes[0])
                        elif 'pB' in line:
                            line = replace_memref(line, kernel_shapes[1])
                            if 'subview' in line:
                                file.write(line)
                                line = next(line_iter, None)
                                line = replace_memref(line, kernel_shapes[1])
                        elif 'pC' in line:
                            line = replace_memref(line, kernel_shapes[2])
                            if 'subview' in line:
                                file.write(line)
                                line = next(line_iter, None)
                                line = replace_memref(line, kernel_shapes[2])
                        else: 
                            # Handle NPU shapes
                            if ('ITout' in line and 'MTin' in line) or ('MTout' in line and 'ITin' in line):
                                # Check that the buffer indexes are the same. This is necessary because the 
                                # shape that's used will be based on the index of the buffer
                                (equal_idxs, idx) = check_buffer_indexes(line)
                                if not equal_idxs:
                                    raise ValueError(f"The buffer indexes are not the same for line: {line}")
                                if idx == -1:
                                    raise ValueError(f"The buffer index is -1 for line: {line}")
                                line = replace_memref(line, (npu_shapes[idx][0], npu_shapes[idx][1] // 4)) # Divide by 4 here since the NPU inputs will be split across 4 rows or 4 columns
                            # Handle kernel shapes
                            else:
                                mtbuf_idx = int(line[line.find('mtbuffer_')+len('mtbuffer_')])
                                if mtbuf_idx == -1:
                                    raise ValueError(f"The buffer index is -1 for line: {line}")
                                line = replace_memref(line, kernel_shapes[mtbuf_idx])
                                if 'subview' in line:
                                    file.write(line)
                                    line = next(line_iter, None)
                                    line = replace_memref(line, kernel_shapes[mtbuf_idx])
                    else:
                        line = replace_func_memref(line, kernel_shapes)
                else:
                    break
            file.write(line)
            line = next(line_iter, None)
        line = replace_sequence_memref(line, npu_shapes) # In the sequence function since the memrefs will be in two dimensions in the generated MLIR
        file.write(line)
        line = next(line_iter, None)
        while line is not None:
            if 'memref' in line and 'itbuffer' in line:
                line = replace_2d_memref(line, npu_shapes)
                shape_idx =  int(line[line.find('itbuffer_')+len('itbuffer_')])
                if shape_idx == 0 or shape_idx == 1:
                    itbuf_idx = int(line[line.find('ITout_')+len('ITout_')])
                    mtbuf_idx = int(line[line.find('MTin_')+len('MTin_')])
                    if itbuf_idx == -1 or mtbuf_idx == -1 or itbuf_idx != mtbuf_idx:
                        raise ValueError(f"The buffer index is incorrect for line: {line}")
                elif shape_idx == 2:
                    itbuf_idx = int(line[line.find('ITin_')+len('ITin_')])
                    if itbuf_idx == -1:
                        raise ValueError(f"The buffer index is -1 for line: {line}")
                else:
                    raise ValueError(f"Invalid shape index: {shape_idx}")
                line = replace_sequence_sizes(line, npu_shapes[shape_idx], itbuf_idx)
            elif 'arith.constant' in line:
                ssa_consts.append(line[line.index('c')+1:line.index('=')].strip())
            file.write(line)
            line = next(line_iter, None)
    # Check if all the SSA constants needed are in the file
    required_ssa_consts_kernel = {f'{dim}' for shape in kernel_shapes for dim in shape}
    required_ssa_consts_npu = {f'{dim}' for shape in npu_shapes for dim in shape}
    required_ssa_consts_offsets = {f'{shape[1]//4*i}' for shape in npu_shapes for i in range(4)}
    missing_ssa_consts_offsets = required_ssa_consts_offsets - set(ssa_consts)
    missing_ssa_consts_kernel = required_ssa_consts_kernel - set(ssa_consts) - missing_ssa_consts_offsets
    missing_ssa_consts_npu = required_ssa_consts_npu - set(ssa_consts) - missing_ssa_consts_kernel - missing_ssa_consts_offsets
    if missing_ssa_consts_kernel or missing_ssa_consts_npu or missing_ssa_consts_offsets:
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
                    for const in missing_ssa_consts_offsets:
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
    total_input_size = sum(shape[0] * shape[1] for shape in kernel_shapes[:-1]) * DTYPE_SIZE[inp_dtype]
    total_output_size = kernel_shapes[-1][0] * kernel_shapes[-1][1] * DTYPE_SIZE[out_dtype]
    # Compute IT to CT data theoretical transfer time. Using the aggreggate BW of one IT to CT
    it_to_ct_transfer_time = (total_input_size / 10e9) / 32 # 32 GB/s
    # Compute the theoretical execution time
    theoretical_exec_time = (kernel_shapes[0][0] * kernel_shapes[0][1] * kernel_shapes[1][1]) / (DTYPE_PERFORMANCE[inp_dtype] * 10e9) # Clock is 1 GHz, only using 1 CT so don't need to divide by number of CTs used
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
        input_file = Path(__file__).parent / EDITED_MLIR_FILE_NAME
        output_file = Path(__file__).parent / workload[DIRECTORY_KEY] / workload[FILE_NAME_KEY]
        inp_dtype = workload[DATA_TYPE_INPUT_KEY]
        out_dtype = workload[DATA_TYPE_OUTPUT_KEY]
        npu_shapes = workload[NPU_SHAPES_KEY]
        kernel_shapes = workload[KERNEL_SHAPES_KEY]
        process_mlir_file(input_file, output_file, inp_dtype, out_dtype, npu_shapes, kernel_shapes)

    # Save the size and intensity of each workload to a data file
    data_file = Path(__file__).parent / 'theoretical_performance.csv'
    with open(data_file, 'w', newline='') as df:
        writer = csv.writer(df)
        writer.writerow([
            "Workload Shapes", "NPU Shapes", "Kernel Shapes", 
            "Data Type Input", "Data Type Output", 
            "NPU Size (KB)", "Kernel Size (KB)", 
            "NPU Intensity (Op/B)", "Kernel Intensity (Op/B)", 
            "Theoretical Kernel Execution Time (us)", 
            "Data Tiles", "Theoretical NPU Execution Time (ms)"
        ])
        for workload in WORKLOADS:
            workload_shapes = workload[WORKLOAD_SHAPES_KEY]
            npu_shapes = workload[NPU_SHAPES_KEY]
            kernel_shapes = workload[KERNEL_SHAPES_KEY]
            inp_dtype = workload[DATA_TYPE_INPUT_KEY]
            out_dtype = workload[DATA_TYPE_OUTPUT_KEY]
            npu_size_kb, kernel_size_kb, npu_intensity, kernel_intensity = calculate_workload_size_and_intensity(npu_shapes, kernel_shapes, inp_dtype, out_dtype)
            theoretical_kernel_exec_time_us = calculate_theoretical_kernel_exec_time_us(kernel_shapes, inp_dtype, out_dtype)
            data_tiles = calculate_data_tiles(workload_shapes, npu_shapes)
            theoretical_npu_exec_time_ms = calculate_theoretical_workload_exec_time_ms(theoretical_kernel_exec_time_us, workload_shapes, npu_shapes, inp_dtype, out_dtype)
            writer.writerow([
                workload_shapes, npu_shapes, kernel_shapes, 
                inp_dtype, out_dtype, 
                f"{npu_size_kb:.2f}", f"{kernel_size_kb:.2f}", 
                f"{npu_intensity:.2f}", f"{kernel_intensity:.2f}", 
                f"{theoretical_kernel_exec_time_us:.2f}", 
                data_tiles, f"{theoretical_npu_exec_time_ms:.2f}"
            ])

if __name__ == '__main__':
    execute()
