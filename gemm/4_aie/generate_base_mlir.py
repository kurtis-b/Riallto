import numpy as np
from pathlib import Path
from npu.build.kernel import Kernel
from npu.build.appbuilder import AppBuilder
from npu.build.itkernel import ITWrite, ITRead
from npu.build.mtkernel import MTSplit, MTConcat
from ml_dtypes import bfloat16
import workloads


class Mmul_4aie(AppBuilder):
    def __init__(self, kernel_0: Kernel, kernel_1: Kernel, kernel_2: Kernel, kernel_3: Kernel):
        self.kernel_list = [kernel_0, kernel_1, kernel_2, kernel_3]
        super().__init__()

    """Callgraph to run 4 GEMM kernels"""
    def callgraph(self, mtx_a: np.ndarray, mtx_b: np.ndarray, mtx_c: np.ndarray) -> None:
        # NOTE: Need to split the B matrix second dimension instead of the A matrix first dimension
        # because one of the workloads gives an A matrix with 8 rows. The kernels need each 
        # A matrix input to have at least 8 rows, so in that case the A matrix must be broadcasted.
        # Splitting the A matrix first dimension would require also giving the same number of 
        # rows for the B matrix. Thus, the B matrix second dimension is split instead.

        # Since it's currently not possible in Riallto to split the matrix in the second dimension, 
        # but possible to just generate binaries by passing the MLIR and kernels to the appxclbinbuilder, 
        # then generate a generic MLIR file that splits the matrix in the first dimension. Using that file,
        # make changes as necessary so that the MLIR instead reflects the split in the second dimension, 
        # and use that file to generate the MLIR files specific to the workload.
        input_b = ITRead(mtx_b)
        input_a = MTSplit(mtx_a, 4)
        output_mtxs = [k(x, input_b)
                  for k, x in zip([self.kernel_list[i] for i in range(4)], input_a)]
        output_mtx = MTConcat(output_mtxs)
        _ = ITWrite(output_mtx, bufref=mtx_c)


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
    # Create an instance of the applicaiton
    kernel_src = Path(__file__).parent.parent / "mmul_start.cc"
    kernel_src = str(kernel_src)
    kernel_0 = generate_kernel_start(kernel_src, (64, 64), (64, 64), np.uint8, np.uint16)
    kernel_1 = generate_kernel_start(kernel_src, (64, 64), (64, 64), np.uint8, np.uint16)
    kernel_2 = generate_kernel_start(kernel_src, (64, 64), (64, 64), np.uint8, np.uint16)
    kernel_3 = generate_kernel_start(kernel_src, (64, 64), (64, 64), np.uint8, np.uint16)
    kernel_0.tloc = (0,5)
    kernel_1.tloc = (0,4)
    kernel_2.tloc = (0,3)
    kernel_3.tloc = (0,2)
    app_builder = Mmul_4aie(kernel_0, kernel_1, kernel_2, kernel_3)

    # Trace the callgraph
    npu_mtxa_shape = (4*64, 64)
    npu_mtxb_shape = (64, 64)
    npu_mtxc_shape = (4*64, 64)
    mtx_a = np.zeros(shape=(npu_mtxa_shape[0], npu_mtxa_shape[1]), dtype=np.uint8)
    mtx_b = np.zeros(shape=(npu_mtxb_shape[0], npu_mtxb_shape[1]), dtype=np.uint8)
    mtx_c = np.zeros(shape=(npu_mtxc_shape[0], npu_mtxc_shape[1]), dtype=np.uint16)

    # Generate the MLIR file
    app_builder.previous_build_args = (mtx_a, mtx_b, mtx_c)
    app_builder.to_mlir(mtx_a, mtx_b, mtx_c, file=f"{Path(__file__).parent / workloads.GENERIC_MLIR_FILE_NAME}")


if __name__ == "__main__":
    execute()
