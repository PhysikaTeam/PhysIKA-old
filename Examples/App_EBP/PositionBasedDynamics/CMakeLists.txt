add_library(PositionBasedDynamics
		 ${PROJECT_PATH}/Common/Common.h
		
		MathFunctions.cpp
		MathFunctions.h
		PositionBasedDynamics.cpp
		PositionBasedDynamics.h
		PositionBasedElasticRods.cpp
		PositionBasedElasticRods.h
		PositionBasedFluids.cpp
		PositionBasedFluids.h
		PositionBasedRigidBodyDynamics.cpp
		PositionBasedRigidBodyDynamics.h
		PositionBasedGenericConstraints.h
		SPHKernels.cpp
		SPHKernels.h
		TimeIntegration.cpp
		TimeIntegration.h
		
		CMakeLists.txt
)
set_target_properties(PositionBasedDynamics PROPERTIES FOLDER "${FOLDER_PREFIX}")

include_directories( ${EigenDir} )

install(TARGETS PositionBasedDynamics
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)




