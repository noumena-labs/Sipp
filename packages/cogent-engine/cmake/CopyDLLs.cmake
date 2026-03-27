# cmake/CopyDLLs.cmake
if(NOT DEFINED src OR NOT DEFINED dst)
  message(FATAL_ERROR "CopyDLLs.cmake requires -Dsrc and -Ddst")
endif()

file(GLOB dlls "${src}/*.dll")
if(NOT dlls)
  message(STATUS "No DLLs found in ${src}")
endif()

file(MAKE_DIRECTORY "${dst}")
foreach(f IN LISTS dlls)
  file(COPY "${f}" DESTINATION "${dst}")
endforeach()
