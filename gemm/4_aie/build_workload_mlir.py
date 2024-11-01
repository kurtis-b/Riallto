from pathlib import Path
import workloads
from workloads import WORKLOADS
from generate_base_mlir import generate_kernel_start, generate_kernel_accum, generate_kernel_end
from ml_dtypes import bfloat16
from npu.build.appxclbinbuilder import AppXclbinBuilder
import numpy as np
import os
import shutil
import re
from pprint import pprint


def execute():
    # Create the xclbin from the kernel source and MLIR file for each workload
    for workload in WORKLOADS:
        # Create an instance of the application
        kernel_src = [Path(__file__).parent.parent / "mmul_start.cc", Path(__file__).parent.parent / "mmul_start.cc", Path(__file__).parent.parent / "mmul_start.cc", Path(__file__).parent.parent / "mmul_start.cc"]
        kernels = []
        for src in kernel_src:
            # Read the kernel source file and replace the data types based on the workload
            kernel_src0 = ''
            with open(src, 'r') as file:
                kernel_src0 = file.read()

                # Determine the datatype from the workload
                inp_datatype = workload[workloads.DATA_TYPE_INPUT_KEY]
                inp_datatype_txt = 'bfloat16' if inp_datatype == 'bfloat16' else 'uint8_t'
                out_datatype = workload[workloads.DATA_TYPE_OUTPUT_KEY]
                out_datatype_txt = 'float' if out_datatype == 'i32' else 'uint16_t'

                # Replace the data types in the kernel source
                kernel_src0 = kernel_src0.replace('uint8_t', inp_datatype_txt)
                kernel_src0 = kernel_src0.replace('uint16_t', out_datatype_txt)
                # Replace the matrix dimensions in the kernel source
                def replace_dimension(line, key, value):
                    pattern = re.compile(rf'const int {key} = \d+;')
                    return pattern.sub(f'const int {key} = {value};', line)
                kernel_src0 = replace_dimension(kernel_src0, 'M', workload[workloads.KERNEL_MMUL_CONFIG_KEY][0])
                kernel_src0 = replace_dimension(kernel_src0, 'K', workload[workloads.KERNEL_MMUL_CONFIG_KEY][1])
                kernel_src0 = replace_dimension(kernel_src0, 'N', workload[workloads.KERNEL_MMUL_CONFIG_KEY][2])
                # Replace the upper bounds in the kernel source
                def replace_ub(line, key, value):
                    pattern = re.compile(rf'const int {key[0]} = \d+ / {key[1]};')
                    return pattern.sub(f'const int {key[0]} = {value} / {key[1]};', line)
                kernel_src0 = replace_ub(kernel_src0, ('rowA', 'M'), workload[workloads.KERNEL_SHAPES_KEY][0][0])
                kernel_src0 = replace_ub(kernel_src0, ('colA', 'K'), workload[workloads.KERNEL_SHAPES_KEY][0][1])
                kernel_src0 = replace_ub(kernel_src0, ('colB', 'N'), workload[workloads.KERNEL_SHAPES_KEY][1][1])
                # Replace the accum tag value in the kernel source
                def replace_accum(line, inp_datatype):
                    value = '32' if inp_datatype == 'uint8' else 'float'
                    pattern = re.compile(rf'using MMUL = ::aie::mmul<M, K, N, {inp_datatype}, {inp_datatype}, acc\d+>;')
                    return pattern.sub(f'using MMUL = ::aie::mmul<M, K, N, {inp_datatype}, {inp_datatype}, acc{value}>;', line)
                kernel_src0 = replace_accum(kernel_src0, workload[workloads.DATA_TYPE_INPUT_KEY])
            temp_kernel_src_path = Path(__file__).parent / "temp_kernel_src.cc"
            with open(temp_kernel_src_path, 'w') as temp_file:
                temp_file.write(kernel_src0)
            inp_dtype = np.uint8 if inp_datatype == 'uint8' else bfloat16
            out_dtype = np.uint16 if out_datatype == 'i16' else np.float32
            if 'start' in str(src):
                kernels.append(generate_kernel_start(str(temp_kernel_src_path), workload[workloads.KERNEL_SHAPES_KEY][0], workload[workloads.KERNEL_SHAPES_KEY][1], inp_dtype, out_dtype))
            elif 'accum' in str(src):
                kernels.append(generate_kernel_accum(str(temp_kernel_src_path), workload[workloads.KERNEL_SHAPES_KEY][0], workload[workloads.KERNEL_SHAPES_KEY][1], inp_dtype, out_dtype))
            else:
                kernels.append(generate_kernel_end(str(temp_kernel_src_path), workload[workloads.KERNEL_SHAPES_KEY][0], workload[workloads.KERNEL_SHAPES_KEY][1], inp_dtype, out_dtype))
            os.remove(temp_kernel_src_path)
        ab = AppXclbinBuilder()
        ab.build(workloads.APP_NAME, Path(__file__).parent / workload[workloads.DIRECTORY_KEY] / workload[workloads.FILE_NAME_KEY], kernels, debug=True)

        # Move the xclbin and seq file to the directory from the workload dict
        workload_dir = Path(__file__).parent / workload[workloads.DIRECTORY_KEY]
        xclbin_file = Path(__file__).parent / f"{workloads.APP_NAME}.xclbin"
        seq_file = Path(__file__).parent / f"{workloads.APP_NAME}.seq"

        if not workload_dir.exists():
            workload_dir.mkdir(parents=True, exist_ok=True)

        shutil.move(str(xclbin_file), str(workload_dir / f"{workloads.APP_NAME}.xclbin"))
        shutil.move(str(seq_file), str(workload_dir / f"{workloads.APP_NAME}.seq"))

        # Rename the xclbin and seq files by appending the workload file name to its current file name up to the ".mlir" string
        new_xclbin_name = f"{workloads.APP_NAME}_{workload[workloads.FILE_NAME_KEY].split('.mlir')[0]}.xclbin"
        new_seq_name = f"{workloads.APP_NAME}_{workload[workloads.FILE_NAME_KEY].split('.mlir')[0]}.seq"
        # Replace the xclbin and seq files in the directory if they exist
        new_xclbin = workload_dir / new_xclbin_name
        new_seq = workload_dir / new_seq_name
        if new_xclbin.exists():
            new_xclbin.unlink()
        if new_seq.exists():
            new_seq.unlink()
        os.rename(workload_dir / f"{workloads.APP_NAME}.xclbin", workload_dir / new_xclbin_name)
        os.rename(workload_dir / f"{workloads.APP_NAME}.seq", workload_dir / new_seq_name)


if __name__ == "__main__":
    execute()
