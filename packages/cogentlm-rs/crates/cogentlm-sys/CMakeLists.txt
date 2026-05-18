cmake_minimum_required(VERSION 3.20)
project(cogentlm_sys C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED COGENTLM_LLAMA_CPP_DIR)
  set(COGENTLM_LLAMA_CPP_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp")
endif()

if(NOT EXISTS "${COGENTLM_LLAMA_CPP_DIR}/include/llama.h")
  message(FATAL_ERROR "Could not find llama.cpp at ${COGENTLM_LLAMA_CPP_DIR}")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build llama.cpp as static libraries" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common utils for chat templates" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Skip llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Skip llama.cpp server" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Skip llama.cpp tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Skip llama.cpp tools by default" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "Disable OpenSSL in native phase-1 build" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "Use static backend linkage" FORCE)

add_subdirectory("${COGENTLM_LLAMA_CPP_DIR}" llama.cpp)

# The mtmd library is C++-only in upstream llama.cpp. Add just that library
# instead of all llama.cpp tools so the Rust sys crate can expose a small C ABI.
find_package(Threads REQUIRED)

set(COGENTLM_MTMD_DIR "${COGENTLM_LLAMA_CPP_DIR}/tools/mtmd")
set(COGENTLM_MTMD_MODEL_DIR "${COGENTLM_MTMD_DIR}/models")
add_library(mtmd STATIC
  "${COGENTLM_MTMD_DIR}/mtmd.cpp"
  "${COGENTLM_MTMD_DIR}/mtmd-audio.cpp"
  "${COGENTLM_MTMD_DIR}/mtmd-image.cpp"
  "${COGENTLM_MTMD_DIR}/mtmd-helper.cpp"
  "${COGENTLM_MTMD_DIR}/clip.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/cogvlm.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/conformer.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/dotsocr.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/gemma4a.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/gemma4v.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/glm4v.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/granite-speech.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/hunyuanocr.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/internvl.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/kimivl.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/kimik25.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/nemotron-v2-vl.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/llama4.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/llava.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/minicpmv.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/mimovl.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/paddleocr.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/pixtral.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/qwen2vl.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/qwen3vl.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/qwen3a.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/step3vl.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/siglip.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/whisper-enc.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/deepseekocr.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/mobilenetv5.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/youtuvl.cpp"
  "${COGENTLM_MTMD_MODEL_DIR}/yasa2.cpp"
)
target_link_libraries(mtmd PUBLIC ggml llama)
target_link_libraries(mtmd PRIVATE Threads::Threads)
target_include_directories(mtmd PUBLIC "${COGENTLM_MTMD_DIR}")
target_include_directories(mtmd PRIVATE
  "${COGENTLM_LLAMA_CPP_DIR}"
  "${COGENTLM_LLAMA_CPP_DIR}/vendor"
)
target_compile_features(mtmd PRIVATE cxx_std_17)

install(TARGETS mtmd ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

add_library(cogent_shim STATIC
  src/cogent_shim.cpp
  include/cogent_shim.h
)

target_include_directories(cogent_shim
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${COGENTLM_LLAMA_CPP_DIR}/common"
    "${COGENTLM_LLAMA_CPP_DIR}/include"
    "${COGENTLM_LLAMA_CPP_DIR}/ggml/include"
    "${COGENTLM_LLAMA_CPP_DIR}/tools/mtmd"
    "${COGENTLM_LLAMA_CPP_DIR}/vendor"
)

foreach(_cogent_backend CUDA METAL VULKAN OPENMP WEBGPU)
  if(GGML_${_cogent_backend})
    target_compile_definitions(cogent_shim PRIVATE GGML_USE_${_cogent_backend})
  endif()
endforeach()

target_link_libraries(cogent_shim PUBLIC llama)
target_link_libraries(cogent_shim PRIVATE llama-common mtmd)

if(MSVC)
  target_compile_options(cogent_shim PRIVATE /utf-8 /EHsc)
  target_compile_options(mtmd PRIVATE /utf-8 /EHsc)
endif()

install(TARGETS cogent_shim ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

# These llama.cpp helper targets are needed by static consumers but are not all
# installed by upstream CMake rules.
foreach(_cogent_target cpp-httplib llama-common-base)
  if(TARGET ${_cogent_target})
    install(TARGETS ${_cogent_target}
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin)
  endif()
endforeach()
