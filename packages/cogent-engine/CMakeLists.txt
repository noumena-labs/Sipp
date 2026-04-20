cmake_minimum_required(VERSION 3.20)
project(CogentEngineWeb CXX)

set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_DEBUG "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE "" CACHE STRING "" FORCE)

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

option(CE_BUILD_WEBGPU_TEST_BACKEND_OPS "Build llama.cpp test-backend-ops for browser-hosted WebGPU execution" OFF)

function(ce_apply_emscripten_cli_target TARGET_NAME EXPORT_NAME)
  if (NOT EMSCRIPTEN OR NOT TARGET ${TARGET_NAME})
    return()
  endif()

  set(_ce_cli_link_options ${CE_WASM_LINK_FLAGS})
  list(APPEND _ce_cli_link_options
      -sINVOKE_RUN=0
      -sALLOW_TABLE_GROWTH=1
      -sMODULARIZE=1
      -sEXPORT_ES6=1
      -sSINGLE_FILE=0
      "-sEXPORT_NAME='${EXPORT_NAME}'"
      "-sEXPORTED_RUNTIME_METHODS=['callMain','FS']")

  if(CE_WASM_USE_JSPI)
    list(APPEND _ce_cli_link_options
        -sJSPI=1
        "-sJSPI_IMPORTS=['emscripten_sleep']"
        "-sJSPI_EXPORTS=['_main']")
  endif()

  target_link_options(${TARGET_NAME} PRIVATE ${_ce_cli_link_options})
  set_target_properties(${TARGET_NAME} PROPERTIES
      OUTPUT_NAME "${TARGET_NAME}"
      SUFFIX ".js"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endfunction()

# llama.cpp subproject
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
set(LLAMA_WASM_MEM64 ${CE_WASM_MEM64} CACHE BOOL "Use wasm64 memory model for cogent-engine" FORCE)
set(GGML_WEBGPU ON CACHE BOOL "Enable WebGPU backend for CogentEngine browser builds" FORCE)
# common provides chat.h / common_chat_msgs_parse_oaicompat used by engine_bridge.cpp.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common for chat-template helpers" FORCE)
if(CE_BUILD_WEBGPU_TEST_BACKEND_OPS)
  set(LLAMA_BUILD_TESTS ON CACHE BOOL "Build llama.cpp tests" FORCE)
  set(LLAMA_OPENSSL OFF CACHE BOOL "Disable OpenSSL for Emscripten test builds" FORCE)
  set(LLAMA_BUILD_HTML OFF CACHE BOOL "Build JS artifacts for browser-hosted wasm tests" FORCE)
endif()
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

if(CE_BUILD_WEBGPU_TEST_BACKEND_OPS)
  if(NOT TARGET test-backend-ops)
    message(FATAL_ERROR "CE_BUILD_WEBGPU_TEST_BACKEND_OPS requires the llama.cpp test-backend-ops target to be available")
  endif()

  ce_apply_emscripten_cli_target(test-backend-ops TestBackendOpsModule)
endif()

# Plugin Sources
set(PLUGIN_SOURCES
    ${CMAKE_SOURCE_DIR}/native/runtime/inference_runtime.cpp
    ${CMAKE_SOURCE_DIR}/native/runtime/request/request_queue.cpp
    ${CMAKE_SOURCE_DIR}/native/runtime/scheduler/batch_planner.cpp
    ${CMAKE_SOURCE_DIR}/native/runtime/scheduler/slot_scheduler.cpp
    ${CMAKE_SOURCE_DIR}/native/runtime/session/prefix_state_cache.cpp
    ${CMAKE_SOURCE_DIR}/native/runtime/session/prefix_cache_policy.cpp
    ${CMAKE_SOURCE_DIR}/native/runtime/session/session_store.cpp
    ${CMAKE_SOURCE_DIR}/native/runtime/llama/llama_batch_builder.cpp
    ${CMAKE_SOURCE_DIR}/native/runtime/llama/llama_utils.cpp
    ${CMAKE_SOURCE_DIR}/native/api/engine_bridge.cpp
    ${CMAKE_SOURCE_DIR}/native/api/wasm_exports.cpp
)

# llama.cpp skips tools/mtmd entirely under EMSCRIPTEN, so compile the
# library sources directly into the browser target and intentionally
# exclude the desktop CLI/debug executables.
set(CE_MTMD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd)
set(CE_MTMD_MODEL_DIR ${CE_MTMD_DIR}/models)
set(CE_MTMD_SOURCES
    ${CE_MTMD_DIR}/mtmd.cpp
    ${CE_MTMD_DIR}/mtmd-audio.cpp
    ${CE_MTMD_DIR}/mtmd-image.cpp
    ${CE_MTMD_DIR}/mtmd-helper.cpp
    ${CE_MTMD_DIR}/clip.cpp
    ${CE_MTMD_MODEL_DIR}/cogvlm.cpp
    ${CE_MTMD_MODEL_DIR}/conformer.cpp
    ${CE_MTMD_MODEL_DIR}/dotsocr.cpp
    ${CE_MTMD_MODEL_DIR}/gemma4a.cpp
    ${CE_MTMD_MODEL_DIR}/gemma4v.cpp
    ${CE_MTMD_MODEL_DIR}/glm4v.cpp
    ${CE_MTMD_MODEL_DIR}/hunyuanocr.cpp
    ${CE_MTMD_MODEL_DIR}/internvl.cpp
    ${CE_MTMD_MODEL_DIR}/kimivl.cpp
    ${CE_MTMD_MODEL_DIR}/kimik25.cpp
    ${CE_MTMD_MODEL_DIR}/nemotron-v2-vl.cpp
    ${CE_MTMD_MODEL_DIR}/llama4.cpp
    ${CE_MTMD_MODEL_DIR}/llava.cpp
    ${CE_MTMD_MODEL_DIR}/minicpmv.cpp
    ${CE_MTMD_MODEL_DIR}/paddleocr.cpp
    ${CE_MTMD_MODEL_DIR}/pixtral.cpp
    ${CE_MTMD_MODEL_DIR}/qwen2vl.cpp
    ${CE_MTMD_MODEL_DIR}/qwen3vl.cpp
    ${CE_MTMD_MODEL_DIR}/qwen3a.cpp
    ${CE_MTMD_MODEL_DIR}/step3vl.cpp
    ${CE_MTMD_MODEL_DIR}/siglip.cpp
    ${CE_MTMD_MODEL_DIR}/whisper-enc.cpp
    ${CE_MTMD_MODEL_DIR}/deepseekocr.cpp
    ${CE_MTMD_MODEL_DIR}/mobilenetv5.cpp
    ${CE_MTMD_MODEL_DIR}/youtuvl.cpp
)

add_executable(CogentEngine ${PLUGIN_SOURCES})

target_sources(CogentEngine PRIVATE ${CE_MTMD_SOURCES})

target_include_directories(CogentEngine PRIVATE
    ${CMAKE_SOURCE_DIR}/native
    ${CE_MTMD_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor
)
target_link_libraries(CogentEngine PRIVATE llama common)

if(NOT MSVC)
    foreach(_ce_mtmd_source IN LISTS CE_MTMD_SOURCES)
        set_source_files_properties(${_ce_mtmd_source} PROPERTIES COMPILE_OPTIONS "-Wno-cast-qual")
    endforeach()
endif()

foreach(_ggml_target ggml ggml-base ggml-cpu ggml-webgpu)
    if(TARGET ${_ggml_target})
        target_link_libraries(CogentEngine PRIVATE ${_ggml_target})
    endif()
endforeach()

set(CE_WASM_EXPORTED_FUNCTIONS
    "['_CE_Init','_CE_InitWithMultimodal','_CE_Close','_CE_EnqueuePrompt','_CE_EnqueuePromptWithMedia','_CE_CancelQueuedRequest','_CE_RunSchedulerTick','_CE_RunSchedulerBurst','_CE_RunSchedulerBurstWithDeadline','_CE_RunRequestStep','_CE_GetCompletedRequestStatus','_CE_DrainCompletedRequestIds','_CE_DrainRuntimeEvents','_CE_GetCompletedRequestOutputSize','_CE_CopyCompletedRequestOutput','_CE_GetCompletedRequestErrorSize','_CE_CopyCompletedRequestError','_CE_GetCompletedRequestRuntimeObservability','_CE_ConsumeCompletedRequest','_CE_FreeString','_CE_GetRuntimeObservability','_CE_ResetRuntimeObservability','_CE_GetBackendObservabilityJson','_CE_GetMediaMarker','_CE_GetChatTemplate','_CE_ApplyChatTemplate','_malloc','_free']"
)

# HEAPU8 is needed for later binary media packing through the wasm bridge.
set(CE_WASM_RUNTIME_METHODS "['ccall','FS','WORKERFS','UTF8ToString','addFunction','removeFunction','getValue','setValue','HEAP32','HEAPF64','HEAPU8']")

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
        # CE_Init, CE_InitWithMultimodal, CE_RunSchedulerTick, CE_RunSchedulerBurst, and
        # CE_RunSchedulerBurstWithDeadline,
        # CE_RunRequestStep may suspend on the browser WebGPU path.
        # CE_GetBackendObservabilityJson can also suspend because querying ggml-webgpu
        # devices may force adapter/device creation under emdawnwebgpu.
        # CE_Close is intentionally not exported as a JSPI function because the
        # TS wrapper calls it synchronously during teardown/reinit.
        "-sJSPI_EXPORTS=['_main_thread','CE_Init','CE_InitWithMultimodal','CE_RunSchedulerTick','CE_RunSchedulerBurst','CE_RunSchedulerBurstWithDeadline','CE_RunRequestStep','CE_GetBackendObservabilityJson']"
    )
endif()

# Link flags for ES Module build
target_link_options(CogentEngine PRIVATE ${CE_WASM_TARGET_LINK_OPTIONS})

set_target_properties(CogentEngine PROPERTIES
    OUTPUT_NAME "CogentEngine"
    SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dist"
)
