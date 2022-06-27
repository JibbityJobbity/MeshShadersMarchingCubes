# Marching Cubes on Mesh Shaders
This project is an implementaiton of the marching cubes algorithm using mesh shaders.
This was built using Vulkan and C++.

## Running
Run compiled binary with the `shaders` folder in the current working directory. Program observes changes of shaders and recompiles them while running. This checking is disabled when the timer is enabled.

Controls are W, A, S, D to move through the scene. Use the mouse to look around, use `ALT + TAB` to switch out of the window while leaving the program running. Pressing `ESC` will close the program.

## Compile-time Configuration
TODO

Check `Constants.h`. Other minute stuff is littered throughout the code.

## Requirements
This should work with any GPU that supports `VK_NV_mesh_shader`, which is pretty much restricted to NVIDIA GPUs in both the Turing and Ampere architectures (RTX 20** and 30**). Tested and works on an RTX 2080 and 3080. Future generations are uncertain, but will very likely support it. Not compatible with AMD hardware or whatever Intel eventually decides to put out. If an extension ever gets released that is compatible, I would very much love to migrate to that.
### Windows
Visual Studio 2022 17.1.4 was used to build this project. Vulkan SDK 1.3.204 is required, any earlier versions *may* run. Any other external dependencies should be checked out as submodules. Make sure to include shader debugging utilities if running this in the Debug configuration. Use the Release build configuration otherwise.
### Linux
**Not supported**. Although a `CMakeLists.txt` file was included, several linker errors on my Arch Linux system. If anyone feels compelled enough to fix it, the effort would be much appreciated. Requires vulkan headers, `glm`, `glfw` and `libshaderc`.
