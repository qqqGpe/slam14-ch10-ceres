cmake_minimum_required(VERSION 2.8)
project(pose_graph)

#set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Eigen
include_directories("/usr/include/eigen3")

# sophus 
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# ceres
find_package(Ceres REQUIRED)

add_executable(pose_graph_ceres pose_graph_ceres.cpp)
target_link_libraries(pose_graph_ceres
        ${CERES_LIBRARIES}
        ${Sophus_LIBRARIES}
        )

add_executable(pose_graph_ceres_quaternion pose_graph_ceres_quaternion.cpp)
target_link_libraries(pose_graph_ceres_quaternion
        ${CERES_LIBRARIES}
        ${Sophus_LIBRARIES}
        )

