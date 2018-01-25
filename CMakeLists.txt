cmake_minimum_required (VERSION 3.2)
#set c++ version and std library version

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -stdlib=libc++ ")
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_COMPILER clang CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER clang++ CACHE STRING "C++ compiler" FORCE)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

project (edit_transfer_interface)

find_package( OpenCV REQUIRED )

include_directories(
  lib/nanogui/ext/eigen/
  lib/nanogui/ext/glfw/include
  lib/nanogui/ext/nanovg/src
  lib/nanogui/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  lib/tapkee/include/
  lib/opencv-3.0.0/include
)

add_subdirectory(lib/tapkee)
add_subdirectory(lib/nanogui)
add_definitions(${NANOGUI_EXTRA_DEFS})


message(STATUS "Status")
message(STATUS ${NANOGUI_EXTRA_DEFS})
message(STATUS ${OPENCV_libs})
message(STATUS ${OpenCV_LIBS})
message(STATUS ${OpenCV_Includes})
message(STATUS ${TAPKEE_SRC_DIR})
message(STATUS ${NANOGUI_EXTRA_LIBS})

message(STATUS "Status DOne")
file(GLOB SOURCES "src/*.cpp")

ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES})

target_link_libraries(
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  nanogui 
  ${NANOGUI_EXTRA_LIBS}
)
