cmake_minimum_required(VERSION 3.20)
project(sipp_sys C CXX)

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

set(SIPP_LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")

if(NOT EXISTS "${SIPP_LLAMA_CPP_DIR}/include/llama.h")
  message(FATAL_ERROR "Could not find llama.cpp at ${SIPP_LLAMA_CPP_DIR}")
endif()

# --- INCLUDE SHARED LOGIC ---
# Using the specific file shown in your VS Code tree
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/llama_mtmd_sources.cmake)

# Native-specific mtmd configurations
find_package(Threads REQUIRED)
target_link_libraries(mtmd PRIVATE Threads::Threads)
install(TARGETS mtmd ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

# Shim setup
add_library(sipp_shim
  native/llama_shim/sipp_shim.cpp
  native/llama_shim/sipp_shim.h
)

target_include_directories(sipp_shim
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/native/llama_shim"
  PRIVATE
    "${SIPP_LLAMA_CPP_DIR}/common"
    "${SIPP_MTMD_DIR}"
    "${SIPP_LLAMA_CPP_DIR}/vendor"
)

foreach(_sipp_backend CUDA METAL VULKAN OPENMP WEBGPU)
  if(GGML_${_sipp_backend})
    target_compile_definitions(sipp_shim PRIVATE GGML_USE_${_sipp_backend})
  endif()
endforeach()

if(GGML_BACKEND_DL)
  target_compile_definitions(sipp_shim PRIVATE GGML_BACKEND_DL)
endif()

target_link_libraries(sipp_shim PUBLIC llama PRIVATE llama-common mtmd)

if(MSVC)
  target_compile_options(sipp_shim PRIVATE /utf-8 /EHsc)
endif()

install(TARGETS sipp_shim ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

foreach(_sipp_target cpp-httplib llama-common-base)
  if(TARGET ${_sipp_target})
    install(TARGETS ${_sipp_target}
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin)
  endif()
endforeach()

if(GGML_BACKEND_DL)
  foreach(_sipp_target llama llama-common ggml ggml-base)
    if(TARGET ${_sipp_target})
      install(TARGETS ${_sipp_target}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
    endif()
  endforeach()
endif()
