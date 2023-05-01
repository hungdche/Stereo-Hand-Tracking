cmake_minimum_required(VERSION 3.10)
project(hand_tracking)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall")

find_package(OpenCV REQUIRED)
message(STATUS "Using OpenCV Version: ${OpenCV_VERSION}")

find_package(Eigen3 REQUIRED)
message(STATUS "Using Eigen3 Version: ${Eigen3_VERSION}")

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
include_directories(include)

set(PROJECT_SRC
    src/bounding_box.cpp
    src/dataset_loader.cpp
    src/hand_modeler.cpp
    src/stereo_matching.cpp
    src/main.cpp
)

set(PROJECT_LIBS
    ${OpenCV_LIBS}
)

add_executable(${PROJECT_NAME} ${PROJECT_SRC})
target_link_libraries(${PROJECT_NAME} ${PROJECT_LIBS})