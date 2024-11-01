DIRECTORY_KEY = 'directory'
FILE_NAME_KEY = 'file_name'
DATA_TYPE_INPUT_KEY = 'data_type_input'
DATA_TYPE_OUTPUT_KEY = 'data_type_output'
NPU_SHAPES_KEY = 'npu_shapes'
KERNEL_SHAPES_KEY = 'kernel_shapes'
WORKLOAD_SHAPES_KEY = 'workload_shapes'
KERNEL_MMUL_CONFIG_KEY = 'mmul_config'
APP_NAME = 'Mmul_4aie'
GENERIC_MLIR_FILE_NAME = f"{APP_NAME}_DO_NOT_USE.mlir"
EDITED_MLIR_FILE_NAME = f"{APP_NAME}_EDITED.mlir"

WORKLOADS = [
    # TODO: Vary the NPU shapes and MMUL config to see how much it affects the application performance
    # For example, with the uint8 1kx1kx1k workload and MMUL config (8, 8, 4), kernel shapes of [(64, 128), (128, 64), (64, 64)] performed twice as fast as [(64, 64), (64, 64), (64, 64)]
    {
        DIRECTORY_KEY: 'uint8',
        FILE_NAME_KEY: '1kx1kx1k.mlir',
        DATA_TYPE_INPUT_KEY: 'uint8',
        DATA_TYPE_OUTPUT_KEY: 'i16',
        NPU_SHAPES_KEY: [(64, 64), (64, 4*64), (64, 4*64)],  # A, B, C
        KERNEL_SHAPES_KEY: [(64, 64), (64, 64), (64, 64)],  # A, B, C
        WORKLOAD_SHAPES_KEY: [(1024, 1024), (1024, 1024), (1024, 1024)],  # A, B, C
        KERNEL_MMUL_CONFIG_KEY: (8, 8, 4)
    },
    {
        DIRECTORY_KEY: 'bfloat16',
        FILE_NAME_KEY: '1kx1kx1k.mlir',
        DATA_TYPE_INPUT_KEY: 'bfloat16',
        DATA_TYPE_OUTPUT_KEY: 'i32',
        NPU_SHAPES_KEY: [(32, 64), (64, 4*32), (32, 4*32)],  # A, B, C
        KERNEL_SHAPES_KEY: [(32, 64), (64, 32), (32, 32)],  # A, B, C
        WORKLOAD_SHAPES_KEY: [(1024, 1024), (1024, 1024), (1024, 1024)],  # A, B, C
        KERNEL_MMUL_CONFIG_KEY: (8, 8, 4)
    },
    {
        DIRECTORY_KEY: 'uint8',
        FILE_NAME_KEY: '8x4kx4k.mlir',
        DATA_TYPE_INPUT_KEY: 'uint8',
        DATA_TYPE_OUTPUT_KEY: 'i16',
        NPU_SHAPES_KEY: [(8, 128), (128, 4*64), (8,4* 64)],  # A, B, C
        KERNEL_SHAPES_KEY: [(8, 128), (128, 64), (8, 64)],  # A, B, C
        WORKLOAD_SHAPES_KEY: [(8, 4096), (4096, 4096), (8, 4096)],  # A, B, C
        KERNEL_MMUL_CONFIG_KEY: (8, 8, 4)
    },
    {
        DIRECTORY_KEY: 'bfloat16',
        FILE_NAME_KEY: '8x4kx4k.mlir',
        DATA_TYPE_INPUT_KEY: 'bfloat16',
        DATA_TYPE_OUTPUT_KEY: 'i32',
        NPU_SHAPES_KEY: [(8, 128), (128, 4*32), (8, 4*32)],  # A, B, C
        KERNEL_SHAPES_KEY: [(8, 128), (128, 32), (8, 32)],  # A, B, C
        WORKLOAD_SHAPES_KEY: [(8, 4096), (4096, 4096), (8, 4096)],  # A, B, C
        KERNEL_MMUL_CONFIG_KEY: (8, 8, 4)
    }
]
