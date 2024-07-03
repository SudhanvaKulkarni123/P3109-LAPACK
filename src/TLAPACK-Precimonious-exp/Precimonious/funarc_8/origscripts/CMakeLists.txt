cmake_minimum_required(VERSION 3.12)
project(funarc_8)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)




# Add the source file to the project
set(SOURCES
    funarc_8.cpp
)

set(PROJECT_NAME funarc_8)


# Add an executable target
add_executable(${PROJECT_NAME} ${SOURCES})


