cmake_minimum_required(VERSION 3.14)
project(CppML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library sources
add_library(cppml STATIC
    src/core/Matrix.cpp
    src/core/Utils.cpp
    src/core/Metrics.cpp
    src/core/Dataset.cpp
    src/models/LR.cpp
    src/models/Logistic_regression.cpp
    src/models/KNN.cpp
)

target_include_directories(cppml PUBLIC include)

# Example
add_executable(linear_reg_example examples/lr_example.cpp)
add_executable(logistic_reg_example examples/logistic_regression_example.cpp)
add_executable(knn_example examples/KNN_example.cpp)
target_link_libraries(linear_reg_example PRIVATE cppml)
target_link_libraries(logistic_reg_example PRIVATE cppml)
target_link_libraries(knn_example PRIVATE cppml)