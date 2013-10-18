# Top-level Makefile for Physika
# This makefile is by no means good! It needs to be replaced when I get better at Makefile.
# Author: Fei Zhu

include Makefile_header

physika: physika_core physika_dependency

physika_core: physika_dependency
	cd ./Physika_Src/Physika_Core; make

.PHONY: physika_dependency
physika_dependency:
	cd ./Physika_Src/Physika_Dependency; make

.PHONY: cleanall
cleanall:
	cd ./Physika_Src/Physika_Core; make cleanall
	cd ./Physika_Src/Physika_Dependency; make cleanall