from npu.build.appxclbinbuilder import AppXclbinBuilder
from workloads import WORKLOADS
from workloads import APP_NAME, DIRECTORY_KEY, FILE_NAME_KEY, KERNEL_SHAPES_KEY, DATA_TYPE_INPUT_KEY, DATA_TYPE_OUTPUT_KEY, KERNEL_MMUL_CONFIG_KEY
from workloads import INP_DATATYPES as inp_datatypes, OUT_DATATYPES as out_datatypes, ACC_TYPES as acc_types, INP_DTYPES as inp_dtypes, OUT_DTYPES as out_dtypes
from ml_dtypes import bfloat16
from npu.build.kernel import Kernel
from pathlib import Path
import numpy as np
import os
import re
import shutil


def generate_kernel_start(kernel_src: str, kernel_mtxa_shape: tuple, kernel_mtxb_shape: tuple, inp_dtype, out_dtype) -> Kernel:
    def fx_behavior(invobj):
        invobj.pA.array = np.ndarray(shape=kernel_mtxa_shape, dtype=inp_dtype)
        invobj.pB.array = np.ndarray(shape=kernel_mtxb_shape, dtype=inp_dtype)
        invobj.pC.array = np.ndarray(shape=(kernel_mtxa_shape[0], kernel_mtxb_shape[1]), dtype=out_dtype)
    return Kernel(kernel_src, behavioralfx=fx_behavior, requires_boilerplate=True)


def generate_kernel_accum(kernel_src: str, kernel_mtxa_shape: tuple, kernel_mtxb_shape: tuple, inp_dtype, out_dtype) -> Kernel:
    def fx_behavior(invobj):
        invobj.pA.array = np.ndarray(shape=kernel_mtxa_shape, dtype=inp_dtype)
        invobj.pB.array = np.ndarray(shape=kernel_mtxb_shape, dtype=inp_dtype)
        invobj.pAccum.array = np.ndarray(shape=(kernel_mtxa_shape[0], kernel_mtxb_shape[1]), dtype=out_dtype)
        invobj.pC.array = np.ndarray(shape=(kernel_mtxa_shape[0], kernel_mtxb_shape[1]), dtype=out_dtype)
    return Kernel(kernel_src, behavioralfx=fx_behavior, requires_boilerplate=True)


def generate_kernel_end(kernel_src: str, kernel_mtxa_shape: tuple, kernel_mtxb_shape: tuple, inp_dtype, out_dtype) -> Kernel:
    def fx_behavior(invobj):
        invobj.pA.array = np.ndarray(shape=kernel_mtxa_shape, dtype=inp_dtype)
        invobj.pB.array = np.ndarray(shape=kernel_mtxb_shape, dtype=inp_dtype)
        invobj.pAccum.array = np.ndarray(shape=(kernel_mtxa_shape[0], kernel_mtxb_shape[1]), dtype=out_dtype)
        invobj.pC.array = np.ndarray(shape=(kernel_mtxa_shape[0], kernel_mtxb_shape[1]), dtype=out_dtype)
    return Kernel(kernel_src, behavioralfx=fx_behavior, requires_boilerplate=True)


def execute():
    # Create the xclbin from the kernel source and MLIR file for each workload
    for workload in WORKLOADS:
        # Create an instance of the application
        kernel_src = [Path(__file__).parent.parent / "mmul_start.cc", Path(__file__).parent.parent / "mmul_accum.cc", Path(__file__).parent.parent / "mmul_accum.cc", Path(__file__).parent.parent / "mmul_end.cc"]
        kernels = []
        for src in kernel_src:
            # Read the kernel source file and replace the data types based on the workload
            kernel_src0 = ''
            with open(src, 'r') as file:
                kernel_src0 = file.read()

                # Determine the datatype from the workload
                inp_datatype = workload[DATA_TYPE_INPUT_KEY]
                inp_datatype_txt = inp_datatypes[inp_datatype]
                out_datatype = workload[DATA_TYPE_OUTPUT_KEY]
                out_datatype_txt = out_datatypes[out_datatype]

                # Replace the data types in the kernel source
                kernel_src0 = kernel_src0.replace('uint8_t', inp_datatype_txt)
                kernel_src0 = kernel_src0.replace('uint16_t', out_datatype_txt)
                # Replace the matrix dimensions in the kernel source
                def replace_dimension(line, key, value):
                    pattern = re.compile(rf'const int {key} = \d+;')
                    return pattern.sub(f'const int {key} = {value};', line)
                kernel_src0 = replace_dimension(kernel_src0, 'M', workload[KERNEL_MMUL_CONFIG_KEY][0])
                kernel_src0 = replace_dimension(kernel_src0, 'K', workload[KERNEL_MMUL_CONFIG_KEY][1])
                kernel_src0 = replace_dimension(kernel_src0, 'N', workload[KERNEL_MMUL_CONFIG_KEY][2])
                # Replace the upper bounds in the kernel source
                def replace_ub(line, key, value):
                    pattern = re.compile(rf'const int {key[0]} = \d+ / {key[1]};')
                    return pattern.sub(f'const int {key[0]} = {value} / {key[1]};', line)
                kernel_src0 = replace_ub(kernel_src0, ('rowA', 'M'), workload[KERNEL_SHAPES_KEY][0][0])
                kernel_src0 = replace_ub(kernel_src0, ('colA', 'K'), workload[KERNEL_SHAPES_KEY][0][1])
                kernel_src0 = replace_ub(kernel_src0, ('colB', 'N'), workload[KERNEL_SHAPES_KEY][1][1])
                # Replace the accum tag value in the kernel source
                def replace_accum(line, out_datatype):
                    value = acc_types[out_datatype]
                    pattern = re.compile(rf'using MMUL = ::aie::mmul<M, K, N, {inp_datatype}, {inp_datatype}, acc\d+>;')
                    return pattern.sub(f'using MMUL = ::aie::mmul<M, K, N, {inp_datatype}, {inp_datatype}, acc{value}>;', line)
                kernel_src0 = replace_accum(kernel_src0, out_datatype)
            temp_kernel_src_path = Path(__file__).parent / "temp_kernel_src.cc"
            with open(temp_kernel_src_path, 'w') as temp_file:
                temp_file.write(kernel_src0)
            inp_dtype = inp_dtypes[inp_datatype]
            out_dtype = out_dtypes[out_datatype]
            if 'start' in str(src):
                kernels.append(generate_kernel_start(str(temp_kernel_src_path), workload[KERNEL_SHAPES_KEY][0], workload[KERNEL_SHAPES_KEY][1], inp_dtype, out_dtype))
            elif 'accum' in str(src):
                kernels.append(generate_kernel_accum(str(temp_kernel_src_path), workload[KERNEL_SHAPES_KEY][0], workload[KERNEL_SHAPES_KEY][1], inp_dtype, out_dtype))
            else:
                kernels.append(generate_kernel_end(str(temp_kernel_src_path), workload[KERNEL_SHAPES_KEY][0], workload[KERNEL_SHAPES_KEY][1], inp_dtype, out_dtype))
            os.remove(temp_kernel_src_path)
        ab = AppXclbinBuilder()
        ab.build(APP_NAME, Path(__file__).parent / workload[DIRECTORY_KEY] / workload[FILE_NAME_KEY], kernels, debug=True)

        # Move the xclbin and seq file to the directory from the workload dict
        workload_dir = Path(__file__).parent / workload[DIRECTORY_KEY]
        xclbin_file = Path(__file__).parent / f"{APP_NAME}.xclbin"
        seq_file = Path(__file__).parent / f"{APP_NAME}.seq"

        if not workload_dir.exists():
            workload_dir.mkdir(parents=True, exist_ok=True)

        shutil.move(str(xclbin_file), str(workload_dir / f"{APP_NAME}.xclbin"))
        shutil.move(str(seq_file), str(workload_dir / f"{APP_NAME}.seq"))

        # Rename the xclbin and seq files by appending the workload file name to its current file name up to the ".mlir" string
        new_xclbin_name = f"{APP_NAME}_{workload[FILE_NAME_KEY].split('.mlir')[0]}.xclbin"
        new_seq_name = f"{APP_NAME}_{workload[FILE_NAME_KEY].split('.mlir')[0]}.seq"
        # Replace the xclbin and seq files in the directory if they exist
        new_xclbin = workload_dir / new_xclbin_name
        new_seq = workload_dir / new_seq_name
        if new_xclbin.exists():
            new_xclbin.unlink()
        if new_seq.exists():
            new_seq.unlink()
        os.rename(workload_dir / f"{APP_NAME}.xclbin", workload_dir / new_xclbin_name)
        os.rename(workload_dir / f"{APP_NAME}.seq", workload_dir / new_seq_name)


if __name__ == "__main__":
    execute()
