# CudaBoids3D

Boids simulation in 3D on the CPU or the GPU (CUDA), rendered with OpenGL.

## Requirements

- CMake 3.24+
- CUDA Toolkit 12.x and a C++17 compiler (MSVC 2022 on Windows, GCC/Clang on Linux)
- An NVIDIA GPU for `-mode GPU` (CPU mode runs without one)
- Linux only: X11 development packages for GLFW, e.g.
  `sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl-dev`

GLFW, GLM and Dear ImGui are downloaded by CMake during configuration (FetchContent),
so they are not stored in this repository. Only the generated glad OpenGL loader lives in `third_party/`.

## Build

```
cmake -B build
cmake --build build --config Release
```

By default the CUDA code is compiled for Turing, Ampere and Ada GPUs (`75;86;89`).
To build only for the GPU in your machine: `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native`.

## Run

```
./build/BoidsApp                  # Linux
.\build\Release\BoidsApp.exe      # Windows
```

Startup arguments:

```
-mode [CPU|GPU]   simulation type (default GPU)
-n [number]       number of boids (default 10000)
```

## Controls

- W/S: camera forward/back
- A/D: camera left/right
- SPACE/LEFT SHIFT: camera up/down
- Right mouse button + mouse movement: rotate the camera
- Mouse scroll: change FOV
- Left mouse button: boids avoid the mouse cursor
- UI sliders change the simulation parameters
