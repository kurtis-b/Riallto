import numpy as np
from pathlib import Path
from npu.build.kernel import Kernel
from npu.build.appbuilder import AppBuilder
from npu.build.itkernel import ITWrite, ITRead
from ml_dtypes import bfloat16
import workloads


class Mmul_1aie(AppBuilder):
    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel
        super().__init__()

    """Callgraph to run 1 GEMM kernel"""
    def callgraph(self, mtx_a: np.ndarray, mtx_b: np.ndarray, mtx_c: np.ndarray) -> None:
        input_a = ITRead(mtx_a)
        input_b = ITRead(mtx_b)
        kernel_output = self.kernel(input_a, input_b)
        _ = ITWrite(kernel_output, bufref=mtx_c)


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
    kernel = generate_kernel_start(kernel_src, (64, 64), (64, 64), np.uint8, np.uint16)
    app_builder = Mmul_1aie(kernel)

    # Trace the callgraph
    npu_mtxa_shape = (64, 64)
    npu_mtxb_shape = (64, 64)
    npu_mtxc_shape = (64, 64)
    mtx_a = np.zeros(shape=(npu_mtxa_shape[0], npu_mtxa_shape[1]), dtype=np.uint8)
    mtx_b = np.zeros(shape=(npu_mtxb_shape[0], npu_mtxb_shape[1]), dtype=np.uint8)
    mtx_c = np.zeros(shape=(npu_mtxc_shape[0], npu_mtxc_shape[1]), dtype=np.uint16)

    # Generate the MLIR file
    app_builder.previous_build_args = (mtx_a, mtx_b, mtx_c)
    app_builder.to_mlir(mtx_a, mtx_b, mtx_c, file=f"{Path(__file__).parent / workloads.GENERIC_MLIR_FILE_NAME}")


if __name__ == "__main__":
    execute()
