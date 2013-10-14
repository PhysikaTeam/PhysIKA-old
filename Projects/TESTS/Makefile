# Makefile for TESTS
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

include $(PHYSIKA_ROOT_DIR)/Makefile_header

TESTS=matrix2x2_test vector2d_test

tests: $(TESTS)

matrix2x2_test: matrix2x2_test.cpp
	$(CXX) $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -L $(PHYSIKA_PUBLIC_DIR)/lib -l matrices $< -o $@

vector2d_test: vector2d_test.cpp
	$(CXX) $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -L $(PHYSIKA_PUBLIC_DIR)/lib -l vectors $< -o $@

.PHONY: clean
clean:
	$(RM) $(TESTS)
