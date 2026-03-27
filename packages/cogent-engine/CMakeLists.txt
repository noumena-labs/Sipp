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
    ${CMAKE_SOURCE_DIR}/native/engine_manager.cpp
    ${CMAKE_SOURCE_DIR}/native/engine_bridge.cpp
    ${CMAKE_SOURCE_DIR}/native/wasm_exports.cpp
)

add_executable(CogentEngine ${PLUGIN_SOURCES})

target_include_directories(CogentEngine PRIVATE ${CMAKE_SOURCE_DIR}/native)
target_link_libraries(CogentEngine PRIVATE llama)

foreach(_ggml_target ggml ggml-base ggml-cpu ggml-webgpu)
    if(TARGET ${_ggml_target})
        target_link_libraries(CogentEngine PRIVATE ${_ggml_target})
    endif()
endforeach()

# Link flags for ES Module build
target_link_options(CogentEngine PRIVATE
    ${CE_WASM_LINK_FLAGS}
    
    # JSPI for async support
    -sJSPI=1
    -sJSPI_IMPORTS=['emscripten_sleep']
    -sJSPI_EXPORTS=['_main_thread','CE_Unity_Init','CE_Unity_Prompt','CE_Unity_PromptAsync','CE_Unity_PlanRoutine','CE_Unity_Close']
    
    # Exported C functions
    -sEXPORTED_FUNCTIONS=['_malloc','_free','_CE_Unity_Init','_CE_Unity_Close','_CE_Unity_Prompt','_CE_Unity_PromptAsync','_CE_Unity_SubmitActions','_CE_Unity_SubmitThings','_CE_Unity_SubmitLocation','_CE_Unity_SubmitGoals','_CE_Unity_SubmitAgentState','_CE_Unity_PlanRoutine','_CE_Unity_PlanRoutineAsync','_CE_Unity_FreePlan','_CE_Unity_FreeString','_CE_Unity_GetVersion','_CE_Unity_IsInitialized','_CE_Unity_HeapAlloc','_CE_Unity_HeapFree','_CE_Unity_SizeOfActionFFI','_CE_Unity_SizeOfGoalFFI','_CE_Unity_SizeOfThingFFI','_CE_Unity_SizeOfAgentStateFFI','_CE_Unity_SizeOfLocationFFI','_CE_Unity_SizeOfPhaseFFI','_CE_Unity_SizeOfExecutionPlanFFI','_CE_Unity_Offset_ActionFFI_Id','_CE_Unity_Offset_ActionFFI_Name','_CE_Unity_Offset_ActionFFI_Description','_CE_Unity_Offset_GoalFFI_Id','_CE_Unity_Offset_GoalFFI_Name','_CE_Unity_Offset_GoalFFI_Description','_CE_Unity_Offset_GoalFFI_Priority','_CE_Unity_Offset_ThingFFI_Id','_CE_Unity_Offset_ThingFFI_Name','_CE_Unity_Offset_ThingFFI_Description','_CE_Unity_Offset_ThingFFI_DistanceToAgent','_CE_Unity_Offset_ThingFFI_Actions','_CE_Unity_Offset_ThingFFI_ActionsCount','_CE_Unity_Offset_AgentStateFFI_Name','_CE_Unity_Offset_AgentStateFFI_Persona','_CE_Unity_Offset_AgentStateFFI_ActiveAction','_CE_Unity_Offset_AgentStateFFI_ActiveThing','_CE_Unity_Offset_LocationFFI_LocationCompressed','_CE_Unity_Offset_PhaseFFI_Step','_CE_Unity_Offset_PhaseFFI_Action','_CE_Unity_Offset_PhaseFFI_Thing','_CE_Unity_Offset_ExecutionPlanFFI_AgentId','_CE_Unity_Offset_ExecutionPlanFFI_Phases','_CE_Unity_Offset_ExecutionPlanFFI_PhasesCount']
    
    # Exported runtime methods
    -sEXPORTED_RUNTIME_METHODS=['ccall','FS','UTF8ToString','stringToUTF8','lengthBytesUTF8','HEAP8','HEAPU8']
    
    -sALLOW_TABLE_GROWTH=1
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME='CogentEngineModule'
    -sSINGLE_FILE=0
)

set_target_properties(CogentEngine PROPERTIES
    OUTPUT_NAME "CogentEngine"
    SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dist"
)
