cmake_minimum_required(VERSION 3.29)
project(KMeans)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-fopenmp")
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# Aggiungi l'eseguibile
add_executable(KMeans main.cpp)

# Add the resources directory to the list of directories to be copied
add_custom_command(TARGET KMeans POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:KMeans>/resources )