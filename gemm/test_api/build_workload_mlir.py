from pathlib import Path
import workloads
from workloads import WORKLOADS
from npu.build.kernel import Kernel
from ml_dtypes import bfloat16
from npu.build.appxclbinbuilder import AppXclbinBuilder
import numpy as np
import os
import shutil
import re
from pprint import pprint


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
    inp_dtypes = {
        'u8': np.uint8,
        'i8': np.int8,
        'bfloat16': bfloat16,
    }
    out_dtypes = {
        'u16': np.uint16,
        'u32': np.uint32,
        'i16': np.int16,
        'i32': np.int32,
        'f32': np.float32,
        'f64': np.float64,
    }
    inp_datatypes = {
        'u8': 'uint8_t',
        'i8': 'int8',
        'bfloat16': 'bfloat16',
    }
    out_datatypes = {
        'u16': 'uint16_t',
        'u32': 'uint32_t',
        'i16': 'int16',
        'i32': 'int32',
        'f32': 'float',
        'f64': 'double',
    }
    acc_types = {
        'u16': '32',
        'u32': '32',
        'i16': '32',
        'i32': '32',
        'f32': 'float',
        'f64': 'float',
    }
    # Reset the successful and failed configs files
    for workload in WORKLOADS:
        workload_dir = Path(__file__).parent / workload[workloads.DIRECTORY_KEY]
        if not workload_dir.exists():
            workload_dir.mkdir(parents=True, exist_ok=True)
        successful_configs_file = workload_dir / "successful_configs.txt"
        failed_configs_file = workload_dir / "failed_configs.txt"
        if successful_configs_file.exists():
            successful_configs_file.unlink()
        if failed_configs_file.exists():
            failed_configs_file.unlink()
    # Create the xclbin from the kernel source and MLIR file for each workload
    for workload in WORKLOADS:
        try:
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
                    inp_datatype_txt = inp_datatypes[inp_datatype]
                    out_datatype = workload[workloads.DATA_TYPE_OUTPUT_KEY]
                    out_datatype_txt = out_datatypes[out_datatype]

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
                    kernels.append(generate_kernel_start(str(temp_kernel_src_path), workload[workloads.KERNEL_SHAPES_KEY][0], workload[workloads.KERNEL_SHAPES_KEY][1], inp_dtype, out_dtype))
                elif 'accum' in str(src):
                    kernels.append(generate_kernel_accum(str(temp_kernel_src_path), workload[workloads.KERNEL_SHAPES_KEY][0], workload[workloads.KERNEL_SHAPES_KEY][1], inp_dtype, out_dtype))
                else:
                    kernels.append(generate_kernel_end(str(temp_kernel_src_path), workload[workloads.KERNEL_SHAPES_KEY][0], workload[workloads.KERNEL_SHAPES_KEY][1], inp_dtype, out_dtype))
                # os.remove(temp_kernel_src_path)
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
            # Save the M, K, N and output data type configurations that successfully ran to a file
            config_file_path = workload_dir / "successful_configs.txt"
            with open(config_file_path, 'a') as config_file:
                config_file.write(f"Workload: {workload[workloads.FILE_NAME_KEY]}, ")
                config_file.write(f"M: {workload[workloads.KERNEL_MMUL_CONFIG_KEY][0]}, ")
                config_file.write(f"K: {workload[workloads.KERNEL_MMUL_CONFIG_KEY][1]}, ")
                config_file.write(f"N: {workload[workloads.KERNEL_MMUL_CONFIG_KEY][2]}, ")
                config_file.write(f"Input Data Type: {inp_datatype_txt}, ")
                config_file.write(f"Output Data Type: {out_datatype_txt}\n")
        except Exception as e:
            config_file_path = workload_dir / "failed_configs.txt"
            with open(config_file_path, 'a') as config_file:
                config_file.write(f"Workload: {workload[workloads.FILE_NAME_KEY]}, ")
                config_file.write(f"M: {workload[workloads.KERNEL_MMUL_CONFIG_KEY][0]}, ")
                config_file.write(f"K: {workload[workloads.KERNEL_MMUL_CONFIG_KEY][1]}, ")
                config_file.write(f"N: {workload[workloads.KERNEL_MMUL_CONFIG_KEY][2]}, ")
                config_file.write(f"Input Data Type: {inp_datatype_txt}, ")
                config_file.write(f"Output Data Type: {out_datatype_txt}\n")
            pass


if __name__ == "__main__":
    execute()
