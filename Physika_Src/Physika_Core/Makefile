# Top-level Makefile for Physika_Core
# This makefile is by no means good! It needs to be replaced when I get better at Makefile.
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

include $(PHYSIKA_ROOT_DIR)/Makefile_header

physika_core: matrices_in_core utilities_in_core eigen_in_core

matrices_in_core: utilities_in_core
	cd ./Matrices; make

utilities_in_core: eigen_in_core
	cd ./Utilities; make

.PHONY: eigen_in_core
eigen_in_core:
	mkdir -p $(PHYSIKA_PUBLIC_DIR)/include/Physika_Core 
	cp -r ./Eigen $(PHYSIKA_PUBLIC_DIR)/include/Physika_Core

.PHONY: cleanall
cleanall:
	cd ./Matrices; make cleanmatrices
	cd ./Utilities; make cleanutilities
	rm -r $(PHYSIKA_PUBLIC_DIR)/include/Physika_Core