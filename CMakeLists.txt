cmake_minimum_required(VERSION 3.18)

project(cuda_boids_3d LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Główne pliki projektu
set(MY_SOURCES
    src/main.cpp
    src/SimulationCPU.cpp
    src/SimulationGPU.cu
    src/glad.c
)

set(IMGUI_SOURCES
    src/imgui/imgui.cpp
    src/imgui/imgui_demo.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_tables.cpp
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_impl_glfw.cpp
    src/imgui/imgui_impl_opengl3.cpp
)


add_executable(BoidsApp 
    ${MY_SOURCES} 
    ${IMGUI_SOURCES}
)

target_include_directories(BoidsApp PRIVATE
    "${CMAKE_SOURCE_DIR}/include"        
    "${CMAKE_SOURCE_DIR}/include/imgui"
)


find_library(GLFW_LIB_PATH NAMES glfw3 glfw3dll PATHS "${CMAKE_SOURCE_DIR}/libs" NO_DEFAULT_PATH)

if(NOT GLFW_LIB_PATH)
    message(FATAL_ERROR "Błąd: Nie znaleziono biblioteki GLFW w folderze libs!")
endif()

target_link_libraries(BoidsApp PRIVATE
    OpenGL::GL
    ${GLFW_LIB_PATH}
    CUDA::cudart 
)

# --- 5. KONFIGURACJA CUDA ---

set_property(TARGET BoidsApp PROPERTY CUDA_ARCHITECTURES native)

target_compile_definitions(BoidsApp PRIVATE _USE_MATH_DEFINES)


file(COPY "${CMAKE_SOURCE_DIR}/shaders" DESTINATION "${CMAKE_BINARY_DIR}")