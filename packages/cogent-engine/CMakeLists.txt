cmake_minimum_required(VERSION 3.20)
project(CogentEngineWeb CXX)

if (NOT EMSCRIPTEN)
  message(FATAL_ERROR "This project must be built with Emscripten!")
endif()

# Language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Emscripten Config (from original cmake/)
include(${CMAKE_SOURCE_DIR}/cmake/EmscriptenConfig.cmake)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_C_SCAN_FOR_MODULES OFF)

add_compile_options(${CE_WASM_COMPILE_FLAGS})

# llama.cpp subproject
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
set(LLAMA_WASM_MEM64 OFF CACHE BOOL "Use wasm32 memory model for cogent-engine" FORCE)
# Ensure ggml selects wasm-specific code paths instead of generic CPU fallback.
if(EMSCRIPTEN)
  set(CMAKE_SYSTEM_PROCESSOR "wasm32")
endif()
add_subdirectory(third_party/llama.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include)

foreach(t ggml ggml-cpu)
  if (TARGET ${t})
    target_compile_options(${t} PRIVATE -UGGML_USE_OPENMP)
  endif()
endforeach()

# Plugin Sources
set(PLUGIN_SOURCES
    ${CMAKE_SOURCE_DIR}/native/runtime/inference_runtime.cpp
    ${CMAKE_SOURCE_DIR}/native/runtime/session/session_store.cpp
    ${CMAKE_SOURCE_DIR}/native/runtime/llama/llama_utils.cpp
    ${CMAKE_SOURCE_DIR}/native/api/engine_bridge.cpp
    ${CMAKE_SOURCE_DIR}/native/api/wasm_exports.cpp
)

add_executable(CogentEngine ${PLUGIN_SOURCES})

target_include_directories(CogentEngine PRIVATE ${CMAKE_SOURCE_DIR}/native)
target_link_libraries(CogentEngine PRIVATE llama)

foreach(_ggml_target ggml ggml-base ggml-cpu ggml-webgpu)
    if(TARGET ${_ggml_target})
        target_link_libraries(CogentEngine PRIVATE ${_ggml_target})
    endif()
endforeach()

set(CE_WASM_EXPORTED_FUNCTIONS
    "['_CE_Init','_CE_Close','_CE_Prompt','_CE_FreeString','_CE_GetLastPromptPerfJson']"
)

set(CE_WASM_RUNTIME_METHODS "['ccall','FS','UTF8ToString']")

set(CE_WASM_TARGET_LINK_OPTIONS
    ${CE_WASM_LINK_FLAGS}
    -sEXPORTED_FUNCTIONS=${CE_WASM_EXPORTED_FUNCTIONS}
    -sEXPORTED_RUNTIME_METHODS=${CE_WASM_RUNTIME_METHODS}
    -sALLOW_TABLE_GROWTH=1
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME='CogentEngineModule'
    -sSINGLE_FILE=0
)

if(CE_WASM_USE_JSPI)
    list(APPEND CE_WASM_TARGET_LINK_OPTIONS
        -sJSPI=1
        -sJSPI_IMPORTS=['emscripten_sleep']
        -sJSPI_EXPORTS=['_main_thread','CE_Init','CE_Prompt','CE_Close']
    )
endif()

# Link flags for ES Module build
target_link_options(CogentEngine PRIVATE ${CE_WASM_TARGET_LINK_OPTIONS})

set_target_properties(CogentEngine PROPERTIES
    OUTPUT_NAME "CogentEngine"
    SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dist"
)
