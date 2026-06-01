cmake_minimum_required(VERSION 3.20)
project(cogentlm_sys C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_OPENSSL OFF CACHE BOOL "Disable OpenSSL in native phase-1 build" FORCE)
if(NOT DEFINED GGML_BACKEND_DL)
  set(GGML_BACKEND_DL OFF CACHE BOOL "Use dynamic ggml backend plugins")
endif()

if(GGML_BACKEND_DL)
  if(APPLE)
    set(CMAKE_BUILD_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH "@loader_path")
  elseif(UNIX)
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
  endif()
endif()

set(COGENTLM_LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp")

if(NOT EXISTS "${COGENTLM_LLAMA_CPP_DIR}/include/llama.h")
  message(FATAL_ERROR "Could not find llama.cpp at ${COGENTLM_LLAMA_CPP_DIR}")
endif()

# --- INCLUDE SHARED LOGIC ---
# Using the specific file shown in your VS Code tree
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/llama_mtmd_sources.cmake)

# Native-specific mtmd configurations
find_package(Threads REQUIRED)
target_link_libraries(mtmd PRIVATE Threads::Threads)
install(TARGETS mtmd ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

# Shim setup
add_library(cogent_shim
  native/llama_shim/cogent_shim.cpp
  native/llama_shim/cogent_shim.h
)

target_include_directories(cogent_shim
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/native/llama_shim"
  PRIVATE
    "${COGENTLM_LLAMA_CPP_DIR}/common"
    "${COGENTLM_MTMD_DIR}"
    "${COGENTLM_LLAMA_CPP_DIR}/vendor"
)

foreach(_cogent_backend CUDA METAL VULKAN OPENMP WEBGPU)
  if(GGML_${_cogent_backend})
    target_compile_definitions(cogent_shim PRIVATE GGML_USE_${_cogent_backend})
  endif()
endforeach()

if(GGML_BACKEND_DL)
  target_compile_definitions(cogent_shim PRIVATE GGML_BACKEND_DL)
endif()

target_link_libraries(cogent_shim PUBLIC llama PRIVATE llama-common mtmd)

if(MSVC)
  target_compile_options(cogent_shim PRIVATE /utf-8 /EHsc)
endif()

install(TARGETS cogent_shim ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

foreach(_cogent_target cpp-httplib llama-common-base)
  if(TARGET ${_cogent_target})
    install(TARGETS ${_cogent_target}
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin)
  endif()
endforeach()

if(GGML_BACKEND_DL)
  foreach(_cogent_target llama llama-common ggml ggml-base)
    if(TARGET ${_cogent_target})
      install(TARGETS ${_cogent_target}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
    endif()
  endforeach()
endif()
