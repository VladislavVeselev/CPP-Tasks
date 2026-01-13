cmake_minimum_required(VERSION 3.29)
project(ContoursProject)

set(CMAKE_CXX_STANDARD 20)

set(OpenCV_DIR "C:/Users/Vladislav/Desktop/opencv/mingw_build/install")


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(task_openCV main.cpp)
set(CMAKE_CXX_FLAGS_DEBUG "-O2 -g")

target_link_libraries(task_openCV ${OpenCV_LIBS})

add_custom_command(TARGET task_openCV POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${OpenCV_DIR}/x64/mingw/lib"
        $<TARGET_FILE_DIR:task_openCV>)
