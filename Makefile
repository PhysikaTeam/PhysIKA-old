# Top-level Makefile for Physika
# This makefile is by no means good! It needs to be replaced when I get better at Makefile.
# Author: Fei Zhu

include Makefile_header

physika: physika_core

physika_core:
	cd ./Physika_Src/Physika_Core; make

.PHONY: cleanall
cleanall:
	cd ./Physika_Src/Physika_Core; make cleanall