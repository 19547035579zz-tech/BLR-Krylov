# - Find ZFP library (with GPU support)
# This module finds ZFP library and headers.
# It sets the following variables:
#   ZFP_FOUND         - True if ZFP is found
#   ZFP_INCLUDE_DIRS  - ZFP header file directory
#   ZFP_LIBRARIES     - ZFP library files
#   ZFP_HAS_GPU       - True if ZFP has GPU (CUDA/HIP) support

find_path(ZFP_INCLUDE_DIR
    NAMES zfp.h
    PATHS /usr/local/include
          /usr/include
          $ENV{ZFP_ROOT}/include
    DOC "ZFP header file directory"
)

find_library(ZFP_LIBRARY
    NAMES zfp libzfp
    PATHS /usr/local/lib
          /usr/lib
          $ENV{ZFP_ROOT}/lib
    DOC "ZFP library file"
)

# Check if ZFP has GPU support (via zfp_cuda.h)
find_path(ZFP_GPU_INCLUDE_DIR
    NAMES zfp_cuda.h
    PATHS /usr/local/include
          /usr/include
          $ENV{ZFP_ROOT}/include
    DOC "ZFP GPU header file directory"
)

# Set output variables
set(ZFP_INCLUDE_DIRS ${ZFP_INCLUDE_DIR})
set(ZFP_LIBRARIES ${ZFP_LIBRARY})
set(ZFP_HAS_GPU FALSE)

# Validate GPU support
if(ZFP_GPU_INCLUDE_DIR)
    set(ZFP_INCLUDE_DIRS ${ZFP_INCLUDE_DIRS} ${ZFP_GPU_INCLUDE_DIR})
    set(ZFP_HAS_GPU TRUE)
    message(STATUS "ZFP GPU support found (header: ${ZFP_GPU_INCLUDE_DIR})")
else()
    message(WARNING "ZFP GPU support not found (missing zfp_cuda.h)")
endif()

# Handle find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZFP
    REQUIRED_VARS ZFP_INCLUDE_DIR ZFP_LIBRARY
    VERSION_VAR ZFP_VERSION
)

# Mark variables as advanced
mark_as_advanced(ZFP_INCLUDE_DIR ZFP_LIBRARY ZFP_GPU_INCLUDE_DIR)