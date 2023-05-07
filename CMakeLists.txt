cmake_minimum_required(VERSION 3.10)
project(hand_tracking)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall")

list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/extern/libtorch")

find_package(OpenCV REQUIRED)
message(STATUS "Using OpenCV Version: ${OpenCV_VERSION}")

find_package(Eigen3 REQUIRED)
message(STATUS "Using Eigen3 Version: ${Eigen3_VERSION}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/depth_estimation
    ${PROJECT_SOURCE_DIR}/include/hand_pose_estimation
)

set(PROJECT_SRC
    src/dataset_loader.cpp
    src/depth_estimation/hand_modeler.cpp
    src/depth_estimation/stereo_matching.cpp
    src/depth_estimation/hand_segmentation.cpp
    src/hand_pose_estimation/depth_projector.cpp
    src/hand_pose_estimation/heatmap_fusion.cpp
)

set(PROJECT_LIBS
    ${OpenCV_LIBS}
    ${TORCH_LIBS}
    Eigen3::Eigen
)

add_library(lib_${PROJECT_NAME} ${PROJECT_SRC})
target_link_libraries(lib_${PROJECT_NAME} PUBLIC ${PROJECT_LIBS})

add_executable(${PROJECT_NAME} src/depth_estimation/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE lib_${PROJECT_NAME})

add_executable(projection_generation src/hand_pose_estimation/projection_generation.cpp)
target_link_libraries(projection_generation PRIVATE lib_${PROJECT_NAME})