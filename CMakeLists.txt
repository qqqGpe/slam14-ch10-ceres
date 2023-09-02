cmake_minimum_required(VERSION 3.5)
project(pose_graph)
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_BUILD_TYPE "Release")

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Ceres REQUIRED)
find_package(fmt REQUIRED)

include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${GTSAM_INCLUDE_DIR})
# ceres
add_executable(pose_graph_ceres pose_graph_ceres.cpp)
target_link_libraries(pose_graph_ceres
        ${CERES_LIBRARIES}
        ${Sophus_LIBRARIES}
        fmt
        )

add_executable(pose_graph_ceres_quaternion pose_graph_ceres_quaternion.cpp)
target_link_libraries(pose_graph_ceres_quaternion
        ${CERES_LIBRARIES}
        ${Sophus_LIBRARIES}
        fmt
        )

add_executable(pose_graph_gtsam pose_graph_gtsam.cpp)
target_link_libraries(pose_graph_gtsam gtsam)

