# Top-level Makefile for Physika_Dynamics
# This makefile is by no means good! It needs to be replaced when I get better at Makefile.
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

include $(PHYSIKA_ROOT_DIR)/Makefile_header

physika_dynamics: constitutive_models_in_dynamics

constitutive_models_in_dynamics:
	cd ./Constitutive_Models; make

.PHONY: cleanall
cleanall:
	cd ./Constitutive_Models; make cleanconstitutivemodels
	-rm -r $(PHYSIKA_PUBLIC_DIR)/include/Physika_Dynamics