cmake_minimum_required(VERSION 3.25)
project(BLR_Krylov LANGUAGES CXX CUDA)

# 1. Basic Configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CUDA_ARCHITECTURES 80 90)  # Target NVIDIA Hopper/Ada architectures
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# 2. Find Dependencies
find_package(CUDAToolkit 12.8 REQUIRED COMPONENTS cudart cusparse cublas)
find_package(ZFP REQUIRED)
if(NOT ZFP_HAS_GPU)
    message(FATAL_ERROR "ZFP must have GPU support (missing zfp_cuda.h)")
endif()

# 3. Include Directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/utils
    ${ZFP_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 4. Build Utilities Static Library
add_library(blr_utils STATIC
    src/utils/zfp_gpu.cu
    src/utils/morton_code.cu
    src/utils/valid_map.cu
)
target_link_libraries(blr_utils PRIVATE
    CUDA::cudart
    ${ZFP_LIBRARIES}
)
set_target_properties(blr_utils PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
    COMPILE_FLAGS "-O3 -lineinfo"
)

# 5. Build Core Modules Static Library
add_library(blr_core STATIC
    src/blr_sparse_preprocess.cu
    src/ca_krylov_basis.cu
    src/fused_spgemm.cu
    src/iter_controller.cu
)
target_link_libraries(blr_core PRIVATE
    blr_utils
    CUDA::cusparse
    CUDA::cublas
)
set_target_properties(blr_core PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
    COMPILE_FLAGS "-O3 -lineinfo"
    CUDA_COMPILE_FLAGS "-O3 -arch=sm_${CUDA_ARCHITECTURES} -lineinfo --expt-extended-lambda"
)

# 6. Build Main Executable
add_executable(blr_krylov src/main.cu)
target_link_libraries(blr_krylov PRIVATE
    blr_core
    blr_utils
)
set_target_properties(blr_krylov PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 7. Installation Configuration (Optional)
install(TARGETS blr_krylov
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/
    DESTINATION include/BLR-Krylov
)
