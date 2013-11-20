# Makefile for Matrices
# This makefile is by no means good! It needs to be replaced when I get better at Makefile.
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

# include top level makefile header
include $(PHYSIKA_ROOT_DIR)/Makefile_header

# headers in this library
MATRICES_HEADERS=matrix_base.h square_matrix.h matrix_2x2.h matrix_3x3.h matrix_MxN.h

# additional headers that this library denpend on
ADDITIONAL_HEADERS=../Utilities/global_config.h ../Utilities/physika_assert.h ../Vectors/vector_2d.h ../Vectors/vector_3d.h ../Vectors/vector_Nd.h

# additional libraries that this library depend on
ADDITIONAL_LIBS=

# target directory of the headers
TARGET_HEADERS_DIR=$(PHYSIKA_PUBLIC_DIR)/include/Physika_Core/Matrices

# copy header files to target include directory
define COPY_HEADERS
mkdir -p $(TARGET_HEADERS_DIR)
cp $(MATRICES_HEADERS) $(TARGET_HEADERS_DIR)
endef

# object files to be compiled for this library
MATRICES_OBJECTS=matrix_2x2.o matrix_3x3.o matrix_MxN.o

# target name of this library
TARGET=libmatrices.a


matrices: $(TARGET)

$(TARGET): $(MATRICES_OBJECTS)
	ar rs $@ $^; cp $(TARGET) $(PHYSIKA_PUBLIC_DIR)/lib/

$(MATRICES_OBJECTS): %.o: %.cpp $(MATRICES_HEADERS) $(ADDITIONAL_HEADERS) copy_headers
	$(CXX) $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -c $< -o $@

copy_headers: 
	$(COPY_HEADERS)

.PHONY: cleanmatrices
cleanmatrices:
	$(RM) $(TARGET) $(MATRICES_OBJECTS) $(PHYSIKA_PUBLIC_DIR)/lib/$(TARGET)
	-rm -r $(TARGET_HEADERS_DIR)
