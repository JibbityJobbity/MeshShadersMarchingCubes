cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_CXX_STANDARD 17)

add_executable (MarchingSquares
	VulkanRenderer.cpp
	MeshShaderHelloTriangleVK.cpp
	FileObserver.cpp
)
target_link_libraries (MarchingSquares LINK_PUBLIC vulkan glfw dl pthread shaderc_combined glslang SPIRV)
target_compile_features(MarchingSquares PUBLIC cxx_std_17)

add_custom_command(TARGET MarchingSquares PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/MarchingSquares/shaders $<TARGET_FILE_DIR:MarchingSquares>/shaders)
#configure_file(shaders/mesh.mesh.glsl shaders/mesh.mesh.glsl COPYONLY)
#configure_file(shaders/plain.frag.glsl shaders/plain.frag.glsl COPYONLY)