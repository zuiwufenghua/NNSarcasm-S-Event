project(MinervaSegs)

include_directories(
  /opt/LibN3L/
  basic
  /opt/mshadow/
)

add_definitions(-DUSE_CUDA=0)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O0" )
ELSE()
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O3" )
ENDIF()
####for   openblas
add_definitions(-DMSHADOW_USE_CUDA=0)
add_definitions(-DMSHADOW_USE_CBLAS=1)
add_definitions(-DMSHADOW_USE_MKL=0)

SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lopenblas -Wl,-rpath,./")
####endfor openblas

####for   cuda
#add_definitions(-DMSHADOW_USE_CUDA=1)
#add_definitions(-DMSHADOW_USE_CBLAS=1)
#add_definitions(-DMSHADOW_USE_MKL=0)

#SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lcudart -lcublas -lcurand" )
#include_directories(
#  $(USE_CUDA_PATH)/include
#)
#LINK_DIRECTORIES($(USE_CUDA_PATH)/lib64)
####endfor cuda

#add_subdirectory(basic)

#aux_source_directory(. DIR_SRCS)

#add_executable(HWordDetector HWordDetector.cpp)
#add_executable(HSWordDetector HSWordDetector.cpp)
#add_executable(CharDetector CharDetector.cpp)
#add_executable(HCharDetector HCharDetector.cpp)
#add_executable(HSCharDetector HSCharDetector.cpp)
add_executable(SEventDetector SEventDetector.cpp)




#target_link_libraries(HWordDetector openblas)
#target_link_libraries(HSWordDetector openblas)
#target_link_libraries(CharDetector openblas)
#target_link_libraries(HCharDetector openblas)
#target_link_libraries(HSCharDetector openblas)
target_link_libraries(SEventDetector openblas)


