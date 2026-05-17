cmake_minimum_required(VERSION 3.15)
project(alpha_go_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use pybind11's new FindPython mode (CMake 3.12+)
set(PYBIND11_FINDPYTHON ON)

# Force CMake to use the specified Python executable
if(DEFINED Python3_EXECUTABLE)
    # Get Python include path from the executable
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
        OUTPUT_VARIABLE _Python3_INCLUDE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        OUTPUT_VARIABLE _Python3_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    message(STATUS "Using Python executable: ${Python3_EXECUTABLE}")
    message(STATUS "Python include dir: ${_Python3_INCLUDE}")
    message(STATUS "Python version: ${_Python3_VERSION}")

    # Set Python3_INCLUDE_DIRS before find_package
    set(Python3_INCLUDE_DIRS ${_Python3_INCLUDE})
endif()

# Find Python - must be before pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Found Python3: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 version: ${Python3_VERSION}")

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Go game library
add_library(go_game STATIC
    go/go_game.cpp
)
target_include_directories(go_game PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# MCTS library
add_library(mcts STATIC
    mcts/mcts.cpp
)
target_include_directories(mcts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcts PUBLIC go_game)

# Python extension module
pybind11_add_module(alpha_go_cpp
    bindings/bindings.cpp
)
# mcts PUBLIC-links go_game (above), so it propagates transitively here.
# Listing go_game again double-links it (ld: "ignoring duplicate libraries").
target_link_libraries(alpha_go_cpp PRIVATE mcts)

# Tests (Catch2)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

add_executable(go_game_test go/go_game_test.cpp)
target_link_libraries(go_game_test PRIVATE go_game Catch2::Catch2WithMain)

# Install
install(TARGETS alpha_go_cpp LIBRARY DESTINATION .)
