# Standardize llama.cpp configurations for both environments
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build llama.cpp as static libraries" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common utils" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Skip llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Skip llama.cpp server" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Skip llama.cpp tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Skip llama.cpp tools by default" FORCE)

# Ensure the parent provided the root llama.cpp path
if(NOT DEFINED COGENTLM_LLAMA_CPP_DIR)
    message(FATAL_ERROR "COGENTLM_LLAMA_CPP_DIR must be defined before including llama_mtmd_sources.cmake")
endif()

add_subdirectory("${COGENTLM_LLAMA_CPP_DIR}" llama.cpp)
include_directories("${COGENTLM_LLAMA_CPP_DIR}/include")
include_directories("${COGENTLM_LLAMA_CPP_DIR}/ggml/include")

# Define MTMD directories standardized for both builds
set(COGENTLM_MTMD_DIR "${COGENTLM_LLAMA_CPP_DIR}/tools/mtmd")
set(COGENTLM_MTMD_MODEL_DIR "${COGENTLM_MTMD_DIR}/models")

set(SHARED_MTMD_SOURCES
    ${COGENTLM_MTMD_DIR}/mtmd.cpp
    ${COGENTLM_MTMD_DIR}/mtmd-audio.cpp
    ${COGENTLM_MTMD_DIR}/mtmd-image.cpp
    ${COGENTLM_MTMD_DIR}/mtmd-helper.cpp
    ${COGENTLM_MTMD_DIR}/clip.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/cogvlm.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/conformer.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/dotsocr.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/gemma4a.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/gemma4v.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/glm4v.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/granite-speech.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/hunyuanvl.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/internvl.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/kimivl.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/kimik25.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/nemotron-v2-vl.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/llama4.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/llava.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/minicpmv.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/mimovl.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/paddleocr.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/pixtral.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/qwen2vl.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/qwen3vl.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/qwen3a.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/step3vl.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/siglip.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/whisper-enc.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/deepseekocr.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/deepseekocr2.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/mobilenetv5.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/youtuvl.cpp
    ${COGENTLM_MTMD_MODEL_DIR}/yasa2.cpp
)

# Build mtmd as a static library for both Wasm and Native
add_library(mtmd STATIC ${SHARED_MTMD_SOURCES})
target_link_libraries(mtmd PUBLIC ggml llama)
target_include_directories(mtmd PUBLIC "${COGENTLM_MTMD_DIR}")
target_include_directories(mtmd PRIVATE
  "${COGENTLM_LLAMA_CPP_DIR}"
  "${COGENTLM_LLAMA_CPP_DIR}/vendor"
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
