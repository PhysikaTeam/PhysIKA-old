add_library(Utils
		FileSystem.h
		Hashmap.h
		IndexedFaceMesh.cpp
		IndexedFaceMesh.h
		IndexedTetMesh.cpp
		IndexedTetMesh.h
		Logger.h
		ObjectArray.h
		OBJLoader.h
		SceneLoader.cpp
		SceneLoader.h
		StringTools.h
		TetGenLoader.cpp
		TetGenLoader.h
		Timing.h
		VolumeIntegration.cpp
		VolumeIntegration.h

		CMakeLists.txt
)
set_target_properties(Utils PROPERTIES FOLDER "${FOLDER_PREFIX}FluidEvolution")

include_directories( ${EigenDir} )

install(TARGETS Utils
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)




