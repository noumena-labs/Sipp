# Standardize llama.cpp configurations for both environments
if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build llama.cpp as static libraries")
endif()
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common utils" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Skip llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Skip llama.cpp server" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Skip llama.cpp tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Skip llama.cpp tools by default" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "Build llama.cpp mtmd library" FORCE)

# Ensure the parent provided the root llama.cpp path
if(NOT DEFINED SIPP_LLAMA_CPP_DIR)
    message(FATAL_ERROR "SIPP_LLAMA_CPP_DIR must be defined before including llama_mtmd_sources.cmake")
endif()

# Keep the mtmd include path available to the Sipp shim. llama.cpp owns the
# mtmd target and its complete source list.
set(SIPP_MTMD_DIR "${SIPP_LLAMA_CPP_DIR}/tools/mtmd")
add_subdirectory("${SIPP_LLAMA_CPP_DIR}" llama.cpp)

if(MSVC)
    target_compile_options(mtmd PRIVATE /utf-8 /EHsc)
endif()

if (MTMD_NO_LOGGING)
    target_compile_definitions(mtmd PRIVATE MTMD_NO_LOGGING)
endif()
