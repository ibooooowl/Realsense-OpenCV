cmake_minimum_required(VERSION 2.8)
set(MY_PROJECT_NAME Realsense-OpenCV)
project(${MY_PROJECT_NAME})

######## Realsense2 Dependency
find_package(realsense2 REQUIRED)
if(realsense2_FOUND)
        include_directories(${realsense2_INCLUDE_DIR})
endif()

######## OpenCV Dependency
set(OpenCV_DIR "/home/user/opencv/build")
option( OpenCV_STATIC OFF )
find_package( OpenCV REQUIRED )
#
## Set Static Link Runtime Library
#if( OpenCV_STATIC )
#        foreach( flag_var
#                CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
#                CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
#                CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
#                CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO )
#                if( ${flag_var} MATCHES "/MD" )
#                        string( REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}" )
#                endif()
#        endforeach()
#endif()

if(OpenCV_FOUND)
        include_directories(${OpenCV_INCLUDE_DIRS})
endif()




######## Complier message
set(CMAKE_BUILD_TYPE Release)
message(STATUS "System: ${CMAKE_SYSTEM}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

######## Set output directories.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build//bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib)

######## Set C++ 11 standard
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
if (COMPILER_SUPPORTS_CXX11)
        if (CMAKE_COMPILER_IS_GNUCXX)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
        else ()
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
        endif ()
else ()
        message(STATUS "Fatal Error: Your compiler doesn't support c++11, please install it!")
endif ()

######## find all the source files.
file(GLOB HEADER_FILES include/*.h)
file(GLOB SOURCE_FILES src/*.cpp)

######## Set executable file name, and add the source files for it.
add_executable(${MY_PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

######## Add Dependency Library
target_link_libraries(${MY_PROJECT_NAME} ${OpenCV_LIBRARIES} ${realsense2_LIBRARY})