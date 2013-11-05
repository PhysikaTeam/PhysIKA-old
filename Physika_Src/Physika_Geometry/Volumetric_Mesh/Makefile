# Makefile for Volumetric Meshes
# This makefile is by no means good! It needs to be replaced when I get better at Makefile.
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

# include top level makefile header
include $(PHYSIKA_ROOT_DIR)/Makefile_header

# headers in this library
VOLUMETRIC_MESH_HEADERS=volumetric_mesh.h tet_mesh.h

# additional headers that this library denpend on
ADDITIONAL_HEADERS=

# target directory of the headers
TARGET_HEADERS_DIR=$(PHYSIKA_PUBLIC_DIR)/include/Physika_Geometry/Volumetric_Mesh

# copy header files to target include directory
define COPY_HEADERS
mkdir -p $(TARGET_HEADERS_DIR)
cp $(VOLUMETRIC_MESH_HEADERS) $(TARGET_HEADERS_DIR)
endef

# object files to be compiled for this library
VOLUMETRIC_MESH_OBJECTS=tet_mesh.o

# target name of this library
TARGET=libvolumetric_mesh.a


volumetric_mesh: $(TARGET)

$(TARGET): $(VOLUMETRIC_MESH_OBJECTS)
	ar rs $@ $^; cp $(TARGET) $(PHYSIKA_PUBLIC_DIR)/lib/

$(VOLUMETRIC_MESH_OBJECTS): %.o: %.cpp $(VOLUMETRIC_MESH_HEADERS) $(ADDITIONAL_HEADERS) copy_headers
	$(CXX) $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -c $< -o $@

copy_headers:
	$(COPY_HEADERS);

.PHONY: cleanvolumetricmesh
cleanconstitutivemodels:
	$(RM) $(TARGET) $(VOLUMETRIC_MESH_OBJECTS) $(PHYSIKA_PUBLIC_DIR)/lib/$(TARGET)
	-rm -r $(TARGET_HEADERS_DIR)
