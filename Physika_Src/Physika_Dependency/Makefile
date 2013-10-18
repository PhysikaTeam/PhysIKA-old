# Top-level Makefile for Physika_Dependency
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

include $(PHYSIKA_ROOT_DIR)/Makefile_header

physika_dependency: physika_eigen

.PHONY: physika_eigen
physika_eigen:
	mkdir -p $(PHYSIKA_PUBLIC_DIR)/include/Physika_Dependency 
	cp -r ./Eigen $(PHYSIKA_PUBLIC_DIR)/include/Physika_Dependency

.PHONY: cleanall
cleanall:
	-rm -r $(PHYSIKA_PUBLIC_DIR)/include/Physika_Dependency