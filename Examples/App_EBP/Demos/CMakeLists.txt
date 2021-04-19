include(Visualization/CMakeLists.txt)

add_definitions(-DPBD_DATA_PATH="../data")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)                                                         
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# set(CMAKE_CUDA_FLAGS "-arch=sm_60") 


include_directories(
	.
	./FluidDemo
	./Utils
	./P2V
	./Simulation
)


add_subdirectory(Simulation)
add_subdirectory(Utils)

add_subdirectory(${PROJECT_PATH}/Demos/FluidDemo)