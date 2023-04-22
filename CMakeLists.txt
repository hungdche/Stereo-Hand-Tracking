cmake_minimum_required(VERSION 3.10)
project(Hand-Tracking)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hand_tracking src/main.cpp)
target_link_libraries(hand_tracking ${OpenCV_LIBS} )