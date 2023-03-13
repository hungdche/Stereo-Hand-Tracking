CMAKE_MINIMUM_REQUIRED(VERSION 3)

project(hand_tracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall")

find_package(ZED 3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} 
${ZED_LIBRARY_DIR}
${OpenCV_LIBRARY_DIRS}
)
