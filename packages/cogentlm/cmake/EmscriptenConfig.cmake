# ======================================================================================
# Emscripten Configuration Module for CogentEngine
# ======================================================================================
# This file centralizes all Emscripten-specific settings to avoid duplication
# and ensure consistency across all WASM targets.
# ======================================================================================

if(NOT EMSCRIPTEN)
    return()
endif()

# ======================================================================================
# Configuration Options (set these before including this file or via CMakeSettings.json)
# ======================================================================================
option(CE_WASM_PTHREADS     "Enable pthreads support in WASM builds"           OFF)
option(CE_WASM_DEBUG        "Enable debug mode (assertions, symbols)"          OFF)
option(CE_WASM_AGGRESSIVE_OPT "Use -Ofast instead of -O3 (may affect precision)" OFF)
option(CE_SUPPRESS_LLAMA_LOGS "Suppress llama.cpp info/debug logs in production" ON)
option(CE_WASM_FILESYSTEM   "Enable Emscripten virtual filesystem support"     ON)
option(CE_WASM_USE_JSPI     "Enable JSPI-based async exports"                  ON)
option(CE_WASM_MEM64        "Enable wasm64 memory model"                       ON)
set(CE_WASM_LTO_MODE "FULL" CACHE STRING "LTO mode for release WASM builds")
set_property(CACHE CE_WASM_LTO_MODE PROPERTY STRINGS OFF THIN FULL)

set(CE_WASM_INITIAL_MEMORY "512MB" CACHE STRING "Initial WASM memory")
set(CE_WASM_MAXIMUM_MEMORY "16384MB" CACHE STRING "Maximum WASM memory")
set(CE_WASM_STACK_SIZE "16MB" CACHE STRING "WASM stack size")
set(CE_WASM_PTHREAD_STACK_SIZE "2MB" CACHE STRING "Default pthread stack size")
set(CE_WASM_PTHREAD_POOL_SIZE "4" CACHE STRING "Pthread pool size")
set(CE_WASM_ENVIRONMENT "web,worker" CACHE STRING "Emscripten environment list")

string(TOUPPER "${CE_WASM_LTO_MODE}" CE_WASM_LTO_MODE)
if(NOT CE_WASM_LTO_MODE STREQUAL "OFF" AND NOT CE_WASM_LTO_MODE STREQUAL "THIN" AND NOT CE_WASM_LTO_MODE STREQUAL "FULL")
    message(FATAL_ERROR "Invalid CE_WASM_LTO_MODE='${CE_WASM_LTO_MODE}'. Use OFF, THIN, or FULL.")
endif()

# ======================================================================================
# GGML Backend Configuration for Emscripten
# ======================================================================================
# Disable native CPU features not available in WASM
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_AMX OFF CACHE BOOL "" FORCE)
set(GGML_AVX OFF CACHE BOOL "" FORCE)
set(GGML_AVX2 OFF CACHE BOOL "" FORCE)
set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
set(GGML_FMA OFF CACHE BOOL "" FORCE)
set(GGML_F16C OFF CACHE BOOL "" FORCE)
set(GGML_SSE3 OFF CACHE BOOL "" FORCE)
set(GGML_SSE41 OFF CACHE BOOL "" FORCE)

# Enable WASM SIMD
set(GGML_WASM_SIMD ON CACHE BOOL "" FORCE)

# ======================================================================================
# Threading Configuration
# ======================================================================================
if(CE_WASM_PTHREADS)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    
    set(_CE_PTHREAD_COMPILE_FLAGS
        -pthread
        -sUSE_PTHREADS=1
    )
    set(_CE_PTHREAD_LINK_FLAGS
        -pthread
        -sUSE_PTHREADS=1
        -sPTHREAD_POOL_SIZE=${CE_WASM_PTHREAD_POOL_SIZE}
        -sDEFAULT_PTHREAD_STACK_SIZE=${CE_WASM_PTHREAD_STACK_SIZE}
        # CRITICAL: Proxy main() to a worker thread so browser main thread stays responsive
        # This prevents WebGPU synchronization from blocking the UI
        -sPROXY_TO_PTHREAD=1
    )
    add_compile_definitions(GGML_PTHREADS=1)
    add_compile_definitions(CE_WASM_PTHREAD_POOL_SIZE=${CE_WASM_PTHREAD_POOL_SIZE})
else()
    set(_CE_PTHREAD_COMPILE_FLAGS)
    set(_CE_PTHREAD_LINK_FLAGS)
    add_compile_definitions(GGML_PTHREADS=0)
    add_compile_definitions(CE_WASM_PTHREAD_POOL_SIZE=1)
endif()

if(CE_WASM_MEM64)
    set(_CE_MEMORY_MODEL_COMPILE_FLAGS -sMEMORY64=1)
    set(_CE_MEMORY_MODEL_LINK_FLAGS -sMEMORY64=1)
else()
    set(_CE_MEMORY_MODEL_COMPILE_FLAGS -sMEMORY64=0)
    set(_CE_MEMORY_MODEL_LINK_FLAGS -sMEMORY64=0)
endif()

if(CE_SUPPRESS_LLAMA_LOGS)
    add_compile_definitions(CE_SUPPRESS_LLAMA_LOGS=1)
endif()

if(CE_WASM_USE_JSPI)
    add_compile_definitions(CE_WASM_USE_JSPI=1)
else()
    add_compile_definitions(CE_WASM_USE_JSPI=0)
endif()

# ======================================================================================
# Optimization Configuration
# ======================================================================================
if(CE_WASM_DEBUG)
    set(_CE_DEBUG_PATH_REMAP_FLAGS)

    if(CMAKE_HOST_WIN32)
        file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" _ce_source_root_cmake)
        file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}" _ce_source_root_native)
        set(_ce_source_root_cmake_lower "${_ce_source_root_cmake}")
        set(_ce_source_root_native_lower "${_ce_source_root_native}")

        if(_ce_source_root_cmake_lower MATCHES "^[A-Z]:")
            string(SUBSTRING "${_ce_source_root_cmake_lower}" 0 1 _ce_source_drive)
            string(TOLOWER "${_ce_source_drive}" _ce_source_drive_lower)
            string(SUBSTRING "${_ce_source_root_cmake_lower}" 1 -1 _ce_source_suffix)
            set(_ce_source_root_cmake_lower "${_ce_source_drive_lower}${_ce_source_suffix}")
        endif()

        if(_ce_source_root_native_lower MATCHES "^[A-Z]:")
            string(SUBSTRING "${_ce_source_root_native_lower}" 0 1 _ce_source_drive)
            string(TOLOWER "${_ce_source_drive}" _ce_source_drive_lower)
            string(SUBSTRING "${_ce_source_root_native_lower}" 1 -1 _ce_source_suffix)
            set(_ce_source_root_native_lower "${_ce_source_drive_lower}${_ce_source_suffix}")
        endif()

        if(NOT _ce_source_root_cmake STREQUAL _ce_source_root_cmake_lower)
            list(APPEND _CE_DEBUG_PATH_REMAP_FLAGS
                "-ffile-prefix-map=${_ce_source_root_cmake}=${_ce_source_root_cmake_lower}"
                "-fdebug-prefix-map=${_ce_source_root_cmake}=${_ce_source_root_cmake_lower}"
                "-fmacro-prefix-map=${_ce_source_root_cmake}=${_ce_source_root_cmake_lower}"
            )
        endif()

        if(NOT _ce_source_root_native STREQUAL _ce_source_root_native_lower)
            list(APPEND _CE_DEBUG_PATH_REMAP_FLAGS
                "-ffile-prefix-map=${_ce_source_root_native}=${_ce_source_root_native_lower}"
                "-fdebug-prefix-map=${_ce_source_root_native}=${_ce_source_root_native_lower}"
                "-fmacro-prefix-map=${_ce_source_root_native}=${_ce_source_root_native_lower}"
            )
        endif()
    endif()

    set(_CE_OPT_FLAGS -O0)
    set(_CE_DEBUG_COMPILE_FLAGS
        -g3
        ${_CE_DEBUG_PATH_REMAP_FLAGS}
    )
    set(_CE_DEBUG_LINK_FLAGS
        # Keep DWARF in the final linked wasm so VS Code can load C/C++ sources.
        -g3
        -sASSERTIONS=2
        -gsource-map
        "--source-map-base=./"
        ${_CE_DEBUG_PATH_REMAP_FLAGS}
    )
    set(_CE_LTO_FLAGS)
    set(_CE_ACTIVE_LTO_MODE OFF)
else()
    if(CE_WASM_AGGRESSIVE_OPT)
        # Equivalent to -Ofast but without deprecated warning and compatible with ggml.
        # -Ofast = -O3 + -ffast-math, but -ffast-math includes -ffinite-math-only which
        # breaks ggml's NaN/Inf checks. So we use -O3 + individual fast-math flags,
        # excluding -ffinite-math-only.
        # See: https://github.com/ggml-org/llama.cpp/pull/7154
        set(_CE_OPT_FLAGS
            -O3
            # Fast-math components (excluding -ffinite-math-only for ggml compatibility)
            #   Excluding -funsafe-math-optimizations and -fassociative-math for f16 correctness
            -fno-math-errno
            -fno-trapping-math
            -freciprocal-math
            -fno-signed-zeros
            -fno-rounding-math
            -ffp-contract=fast
            # Other
            -DNDEBUG
        )
    else()
        set(_CE_OPT_FLAGS -O3 -DNDEBUG)
    endif()
    set(_CE_DEBUG_COMPILE_FLAGS -g0)
    set(_CE_DEBUG_LINK_FLAGS -g0)
    if(CE_WASM_LTO_MODE STREQUAL "OFF")
        set(_CE_LTO_FLAGS)
    elseif(CE_WASM_LTO_MODE STREQUAL "THIN")
        set(_CE_LTO_FLAGS -flto=thin)
    else()
        set(_CE_LTO_FLAGS -flto=full)
    endif()
    set(_CE_ACTIVE_LTO_MODE ${CE_WASM_LTO_MODE})
endif()

# ======================================================================================
# Common Emscripten Compile Flags
# ======================================================================================
# These are compile-time only flags
set(CE_WASM_COMPILE_FLAGS
    # Optimization
    ${_CE_OPT_FLAGS}
    ${_CE_DEBUG_COMPILE_FLAGS}
    
    # Threading
    ${_CE_PTHREAD_COMPILE_FLAGS}
    
    # WebAssembly features
    -msimd128
    -fwasm-exceptions
    -mbulk-memory
    -mnontrapping-fptoint
    ${_CE_MEMORY_MODEL_COMPILE_FLAGS}
    
    # RTTI and LTO
    -frtti
    ${_CE_LTO_FLAGS}
)

# ======================================================================================
# Common Emscripten Link Flags
# ======================================================================================
# These are link-time flags (many -s options only matter at link time)
set(CE_WASM_LINK_FLAGS
    # Optimization (must match compile)
    ${_CE_OPT_FLAGS}
    ${_CE_DEBUG_LINK_FLAGS}
    
    # Threading
    ${_CE_PTHREAD_LINK_FLAGS}
    
    # Exception handling
    -fwasm-exceptions
    
    # LTO
    ${_CE_LTO_FLAGS}
    
    # Environment & Memory
    -sENVIRONMENT=${CE_WASM_ENVIRONMENT}
    -sWASM=1
    -sWASM_BIGINT=1
    ${_CE_MEMORY_MODEL_LINK_FLAGS}
    -sSUPPORT_LONGJMP=wasm
    -sINITIAL_MEMORY=${CE_WASM_INITIAL_MEMORY}
    -sMAXIMUM_MEMORY=${CE_WASM_MAXIMUM_MEMORY}
    -sALLOW_MEMORY_GROWTH=1
    -sSTACK_SIZE=${CE_WASM_STACK_SIZE}
    
    # Runtime
    -sNO_EXIT_RUNTIME=1
    
    # WebGPU support
    -sOFFSCREENCANVAS_SUPPORT=1
)

if(CE_WASM_FILESYSTEM)
    list(APPEND CE_WASM_LINK_FLAGS -sFORCE_FILESYSTEM=1 -lworkerfs.js)
endif()

if(CE_WASM_DEBUG)
    list(APPEND CE_WASM_LINK_FLAGS -sSTACK_OVERFLOW_CHECK=2)
else()
    list(APPEND CE_WASM_LINK_FLAGS -sSTACK_OVERFLOW_CHECK=0)
endif()

# ======================================================================================
# WebGPU Port Configuration
# ======================================================================================
if(NOT EMDAWNWEBGPU_DIR)
    set(_ce_emdawnwebgpu_candidates
        "${CMAKE_SOURCE_DIR}/../Libs/emdawnwebgpu_pkg"
    )

    if(DEFINED EMSCRIPTEN_ROOT_PATH AND EMSCRIPTEN_ROOT_PATH)
        list(APPEND _ce_emdawnwebgpu_candidates
            "${EMSCRIPTEN_ROOT_PATH}/cache/ports/emdawnwebgpu/emdawnwebgpu_pkg"
        )
    endif()

    if(DEFINED EMSDK AND EMSDK)
        list(APPEND _ce_emdawnwebgpu_candidates
            "${EMSDK}/upstream/emscripten/cache/ports/emdawnwebgpu/emdawnwebgpu_pkg"
        )
    endif()

    foreach(_ce_emdawnwebgpu_candidate IN LISTS _ce_emdawnwebgpu_candidates)
        if(EXISTS "${_ce_emdawnwebgpu_candidate}/emdawnwebgpu.port.py")
            set(EMDAWNWEBGPU_DIR "${_ce_emdawnwebgpu_candidate}" CACHE PATH "Path to the emdawnwebgpu port package" FORCE)
            break()
        endif()
    endforeach()
endif()

if(EMDAWNWEBGPU_DIR)
    set(CE_WEBGPU_PORT "--use-port=${EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py")
else()
    set(CE_WEBGPU_PORT "--use-port=emdawnwebgpu")
endif()

# The ggml-webgpu target injects the emdawnwebgpu port itself. Keep discovery here
# for status output and downstream checks, but do not append the port globally or
# Emscripten will reject the duplicated port registration.

# ======================================================================================
# Status Output
# ======================================================================================
message(STATUS "")
message(STATUS "=== CogentEngine Emscripten Configuration ===")
message(STATUS "  Pthreads:           ${CE_WASM_PTHREADS}")
message(STATUS "  Debug mode:         ${CE_WASM_DEBUG}")
message(STATUS "  Aggressive opt:     ${CE_WASM_AGGRESSIVE_OPT}")
message(STATUS "  LTO mode:           ${_CE_ACTIVE_LTO_MODE}")
message(STATUS "  Filesystem:         ${CE_WASM_FILESYSTEM}")
message(STATUS "  JSPI:               ${CE_WASM_USE_JSPI}")
message(STATUS "  Memory64:           ${CE_WASM_MEM64}")
message(STATUS "  Environment:        ${CE_WASM_ENVIRONMENT}")
message(STATUS "  Initial memory:     ${CE_WASM_INITIAL_MEMORY}")
message(STATUS "  Maximum memory:     ${CE_WASM_MAXIMUM_MEMORY}")
message(STATUS "  Stack size:         ${CE_WASM_STACK_SIZE}")
if(CE_WASM_PTHREADS)
    message(STATUS "  Pthread pool:       ${CE_WASM_PTHREAD_POOL_SIZE}")
    message(STATUS "  Pthread stack:      ${CE_WASM_PTHREAD_STACK_SIZE}")
endif()
message(STATUS "  WebGPU port:        ${CE_WEBGPU_PORT}")
message(STATUS "")
