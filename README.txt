# CudaBoids3D
Boids simulation using CPU or GPU (CUDA)

# Build commands:
cmake -B build
cmake --build build --config Release
.\build\Release\BoidsApp.exe

# Startup arguments:
-mode [CPU|GPU] # simulation type (default GPU)
-n [number] # number of boids (default 10000)

# Keybindings:
W/S - Front/Back camera movement (Z axis)
A/D - Left/Right camera movement (X axis)
SPACE/LSHIFT - Up/Down camera movement (Y axis)

Mouse movement + right mouse button - change camera angle
Mouse scroll - change fov
Mouse movement + left mouse button - interaction bois with mouse (they will avoid it)

UI - sliders that allow to change some simulation parameters