import numpy as np
from ml_dtypes import bfloat16

DIRECTORY_KEY = 'directory'
FILE_NAME_KEY = 'file_name'
DATA_TYPE_INPUT_KEY = 'data_type_input'
DATA_TYPE_OUTPUT_KEY = 'data_type_output'
NPU_SHAPES_KEY = 'npu_shapes'
KERNEL_SHAPES_KEY = 'kernel_shapes'
WORKLOAD_SHAPES_KEY = 'workload_shapes'
KERNEL_MMUL_CONFIG_KEY = 'mmul_config'
APP_NAME = 'Mmul_16aie'
GENERIC_MLIR_FILE_NAME = f"{APP_NAME}_DO_NOT_USE.mlir"
EDITED_MLIR_FILE_NAME = f"{APP_NAME}_EDITED.mlir"


INP_DTYPES = {
    'u8': np.uint8,
    'i8': np.int8,
    'bfloat16': bfloat16,
}
OUT_DTYPES = {
    'u16': np.uint16,
    'u32': np.uint32,
    'i16': np.int16,
    'i32': np.int32,
    'f32': np.float32,
    'f64': np.float64,
}
INP_DATATYPES = {
    'u8': 'uint8_t',
    'i8': 'int8',
    'bfloat16': 'bfloat16',
}
OUT_DATATYPES = {
    'u16': 'uint16_t',
    'u32': 'uint32_t',
    'i16': 'int16',
    'i32': 'int32',
    'f32': 'float',
    'f64': 'double',
}
ACC_TYPES = {
    'u16': '32',
    'u32': '32',
    'i16': '32',
    'i32': '32',
    'f32': 'float',
    'f64': 'float',
}

WORKLOADS = [
    # TODO: Vary the NPU shapes and MMUL config to see how much it affects the application performance
    {
        DIRECTORY_KEY: 'uint8',
        FILE_NAME_KEY: '1kx1kx1k.mlir',
        DATA_TYPE_INPUT_KEY: 'i8',
        DATA_TYPE_OUTPUT_KEY: 'i32',
        NPU_SHAPES_KEY: [(64, 4*64), (4*64, 4*64), (64, 4*64)],  # A, B, C
        KERNEL_SHAPES_KEY: [(64, 64), (64, 64), (64, 64)],  # A, B, C
        WORKLOAD_SHAPES_KEY: [(1024, 1024), (1024, 1024), (1024, 1024)],  # A, B, C
        KERNEL_MMUL_CONFIG_KEY: (2, 8, 8)
    },
    {
        DIRECTORY_KEY: 'bfloat16',
        FILE_NAME_KEY: '1kx1kx1k.mlir',
        DATA_TYPE_INPUT_KEY: 'bfloat16',
        DATA_TYPE_OUTPUT_KEY: 'f32',
        NPU_SHAPES_KEY: [(32, 4*32), (4*32, 4*32), (32, 4*32)],  # A, B, C
        KERNEL_SHAPES_KEY: [(32, 32), (32, 32), (32, 32)],  # A, B, C
        WORKLOAD_SHAPES_KEY: [(1024, 1024), (1024, 1024), (1024, 1024)],  # A, B, C
        KERNEL_MMUL_CONFIG_KEY: (4, 8, 4)
    },
    {
        DIRECTORY_KEY: 'uint8',
        FILE_NAME_KEY: '8x4kx4k.mlir',
        DATA_TYPE_INPUT_KEY: 'i8',
        DATA_TYPE_OUTPUT_KEY: 'i32',
        NPU_SHAPES_KEY: [(8, 4*256), (4*256, 4*64), (8,4*64)],  # A, B, C
        KERNEL_SHAPES_KEY: [(8, 256), (256, 64), (8, 64)],  # A, B, C
        WORKLOAD_SHAPES_KEY: [(8, 4096), (4096, 4096), (8, 4096)],  # A, B, C
        KERNEL_MMUL_CONFIG_KEY: (2, 8, 8)
    },
    {
        DIRECTORY_KEY: 'bfloat16',
        FILE_NAME_KEY: '8x4kx4k.mlir',
        DATA_TYPE_INPUT_KEY: 'bfloat16',
        DATA_TYPE_OUTPUT_KEY: 'f32',
        NPU_SHAPES_KEY: [(8, 4*256), (4*256, 4*32), (8, 4*32)],  # A, B, C
        KERNEL_SHAPES_KEY: [(8, 256), (256, 32), (8, 32)],  # A, B, C
        WORKLOAD_SHAPES_KEY: [(8, 4096), (4096, 4096), (8, 4096)],  # A, B, C
        KERNEL_MMUL_CONFIG_KEY: (4, 8, 4)
    }
]
