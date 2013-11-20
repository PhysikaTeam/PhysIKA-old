# Makefile for TESTS
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

include $(PHYSIKA_ROOT_DIR)/Makefile_header

TESTS=matrix2x2_test vector2d_test matrix3x3_test vector3d_test matrixMxN_test constitutive_models_test volumetric_mesh_test

tests: $(TESTS)

matrix2x2_test: matrix2x2_test.cpp $(PHYSIKA_PUBLIC_DIR)/lib/libmatrices.a
	$(CXX) $< -o $@ $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -L $(PHYSIKA_PUBLIC_DIR)/lib -l matrices -l vectors

vector2d_test: vector2d_test.cpp $(PHYSIKA_PUBLIC_DIR)/lib/libvectors.a
	$(CXX) $< -o $@ $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -L $(PHYSIKA_PUBLIC_DIR)/lib -l vectors 

matrix3x3_test: matrix3x3_test.cpp $(PHYSIKA_PUBLIC_DIR)/lib/libmatrices.a
	$(CXX) $< -o $@ $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -L $(PHYSIKA_PUBLIC_DIR)/lib -l matrices -l vectors

vector3d_test: vector3d_test.cpp $(PHYSIKA_PUBLIC_DIR)/lib/libvectors.a
	$(CXX) $< -o $@ $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -L $(PHYSIKA_PUBLIC_DIR)/lib -l vectors 

matrixMxN_test: matrixMxN_test.cpp $(PHYSIKA_PUBLIC_DIR)/lib/libmatrices.a
	$(CXX) $< -o $@ $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -L $(PHYSIKA_PUBLIC_DIR)/lib -l matrices -l vectors

constitutive_models_test: constitutive_models_test.cpp $(PHYSIKA_PUBLIC_DIR)/lib/libmatrices.a $(PHYSIKA_PUBLIC_DIR)/lib/libconstitutive_models.a
	$(CXX) $< -o $@ $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -L $(PHYSIKA_PUBLIC_DIR)/lib -l matrices -l constitutive_models -l vectors

volumetric_mesh_test: volumetric_mesh_test.cpp $(PHYSIKA_PUBLIC_DIR)/lib/libvectors.a $(PHYSIKA_PUBLIC_DIR)/lib/libvolumetric_mesh.a
	$(CXX) $< -o $@ $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -L $(PHYSIKA_PUBLIC_DIR)/lib -l vectors -l volumetric_mesh -l matrices

.PHONY: clean
clean:
	$(RM) $(TESTS)
