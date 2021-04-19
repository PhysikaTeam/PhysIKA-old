add_library(Simulation
		AABB.h
		CollisionDetection.cpp
		CollisionDetection.h
		Constraints.cpp
		Constraints.h
		CubicSDFCollisionDetection.cpp
		CubicSDFCollisionDetection.h
		DistanceFieldCollisionDetection.cpp
		DistanceFieldCollisionDetection.h
		IDFactory.cpp
		IDFactory.h
		LineModel.cpp
		LineModel.h
		NeighborhoodSearchSpatialHashing.cpp
		NeighborhoodSearchSpatialHashing.h
		ParticleData.h
		RigidBody.h
		RigidBodyGeometry.cpp
		RigidBodyGeometry.h
		SimulationModel.cpp
		SimulationModel.h
		TetModel.cpp
		TetModel.h
		TimeManager.cpp
		TimeManager.h
		TimeStepController.cpp
		TimeStepController.h
		TriangleModel.cpp
		TriangleModel.h
		
		BoundingSphere.h
		BoundingSphereHierarchy.cpp
		BoundingSphereHierarchy.h
		kdTree.h
		kdTree.inl

		CMakeLists.txt
)
set_target_properties(Simulation PROPERTIES FOLDER "${FOLDER_PREFIX}FluidEvolution")

############################################################
# Discregrid
############################################################
include_directories(${ExternalInstallDir}/Discregrid/include)
set(SIMULATION_DEPENDENCIES ${SIMULATION_DEPENDENCIES} ExternalProject_Discregrid)
add_dependencies(Simulation ${SIMULATION_DEPENDENCIES})

include_directories( ${EigenDir} )
include_directories(${PROJECT_PATH}/extern/Miniball)

target_link_libraries(Simulation PositionBasedDynamics)


install(TARGETS Simulation
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)