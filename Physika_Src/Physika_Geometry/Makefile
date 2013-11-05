# Top-level Makefile for Physika_Geometry
# This makefile is by no means good! It needs to be replaced when I get better at Makefile.
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

include $(PHYSIKA_ROOT_DIR)/Makefile_header

physika_geometry: volumetric_mesh_in_geometry

volumetric_mesh_in_geometry:
	cd ./Volumetric_Mesh; make

.PHONY: cleanall
cleanall:
	cd ./Volumetric_Mesh; make cleanvolumetricmesh
	-rm -r $(PHYSIKA_PUBLIC_DIR)/include/Physika_Geometry