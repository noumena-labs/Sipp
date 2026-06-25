# Standardize llama.cpp configurations for both environments
if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build llama.cpp as static libraries")
endif()
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common utils" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Skip llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Skip llama.cpp server" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Skip llama.cpp tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Skip llama.cpp tools by default" FORCE)

# Ensure the parent provided the root llama.cpp path
if(NOT DEFINED SIPP_LLAMA_CPP_DIR)
    message(FATAL_ERROR "SIPP_LLAMA_CPP_DIR must be defined before including llama_mtmd_sources.cmake")
endif()

add_subdirectory("${SIPP_LLAMA_CPP_DIR}" llama.cpp)
include_directories("${SIPP_LLAMA_CPP_DIR}/include")
include_directories("${SIPP_LLAMA_CPP_DIR}/ggml/include")

# Define MTMD directories standardized for both builds
set(SIPP_MTMD_DIR "${SIPP_LLAMA_CPP_DIR}/tools/mtmd")
set(SIPP_MTMD_MODEL_DIR "${SIPP_MTMD_DIR}/models")

set(SHARED_MTMD_SOURCES
    ${SIPP_MTMD_DIR}/mtmd.cpp
    ${SIPP_MTMD_DIR}/mtmd-audio.cpp
    ${SIPP_MTMD_DIR}/mtmd-image.cpp
    ${SIPP_MTMD_DIR}/mtmd-helper.cpp
    ${SIPP_MTMD_DIR}/clip.cpp
    ${SIPP_MTMD_MODEL_DIR}/cogvlm.cpp
    ${SIPP_MTMD_MODEL_DIR}/conformer.cpp
    ${SIPP_MTMD_MODEL_DIR}/dotsocr.cpp
    ${SIPP_MTMD_MODEL_DIR}/exaone4_5.cpp
    ${SIPP_MTMD_MODEL_DIR}/gemma4a.cpp
    ${SIPP_MTMD_MODEL_DIR}/gemma4v.cpp
    ${SIPP_MTMD_MODEL_DIR}/gemma4ua.cpp
    ${SIPP_MTMD_MODEL_DIR}/gemma4uv.cpp
    ${SIPP_MTMD_MODEL_DIR}/glm4v.cpp
    ${SIPP_MTMD_MODEL_DIR}/granite-speech.cpp
    ${SIPP_MTMD_MODEL_DIR}/granite4-vision.cpp
    ${SIPP_MTMD_MODEL_DIR}/hunyuanvl.cpp
    ${SIPP_MTMD_MODEL_DIR}/internvl.cpp
    ${SIPP_MTMD_MODEL_DIR}/kimivl.cpp
    ${SIPP_MTMD_MODEL_DIR}/kimik25.cpp
    ${SIPP_MTMD_MODEL_DIR}/nemotron-v2-vl.cpp
    ${SIPP_MTMD_MODEL_DIR}/llama4.cpp
    ${SIPP_MTMD_MODEL_DIR}/llava.cpp
    ${SIPP_MTMD_MODEL_DIR}/minicpmv.cpp
    ${SIPP_MTMD_MODEL_DIR}/mimovl.cpp
    ${SIPP_MTMD_MODEL_DIR}/paddleocr.cpp
    ${SIPP_MTMD_MODEL_DIR}/pixtral.cpp
    ${SIPP_MTMD_MODEL_DIR}/qwen2vl.cpp
    ${SIPP_MTMD_MODEL_DIR}/qwen3vl.cpp
    ${SIPP_MTMD_MODEL_DIR}/qwen3a.cpp
    ${SIPP_MTMD_MODEL_DIR}/step3vl.cpp
    ${SIPP_MTMD_MODEL_DIR}/siglip.cpp
    ${SIPP_MTMD_MODEL_DIR}/whisper-enc.cpp
    ${SIPP_MTMD_MODEL_DIR}/deepseekocr.cpp
    ${SIPP_MTMD_MODEL_DIR}/deepseekocr2.cpp
    ${SIPP_MTMD_MODEL_DIR}/mobilenetv5.cpp
    ${SIPP_MTMD_MODEL_DIR}/youtuvl.cpp
    ${SIPP_MTMD_MODEL_DIR}/yasa2.cpp
)

# Build mtmd as a library for both Wasm and Native
add_library(mtmd ${SHARED_MTMD_SOURCES})
target_link_libraries(mtmd PUBLIC ggml llama)
target_include_directories(mtmd PUBLIC "${SIPP_MTMD_DIR}")
target_include_directories(mtmd PRIVATE
  "${SIPP_LLAMA_CPP_DIR}"
  "${SIPP_LLAMA_CPP_DIR}/vendor"
)
target_compile_features(mtmd PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(mtmd PRIVATE /utf-8 /EHsc)
else()
    foreach(_mtmd_source IN LISTS SHARED_MTMD_SOURCES)
        set_source_files_properties(${_mtmd_source} PROPERTIES COMPILE_OPTIONS "-Wno-cast-qual")
    endforeach()
endif()

if (MTMD_NO_LOGGING)
    target_compile_definitions(mtmd PRIVATE MTMD_NO_LOGGING)
endif()
