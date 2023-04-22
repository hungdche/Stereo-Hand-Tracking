cmake_minimum_required(VERSION 3.10)
project(hand_tracking)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)

set(PROJECT_SRC
    src/dataset_loader.cpp
    src/main.cpp
)

set(PROJECT_LIBS
    ${OpenCV_LIBS}
)

add_executable(${PROJECT_NAME} ${PROJECT_SRC})
target_link_libraries(${PROJECT_NAME} ${PROJECT_LIBS})