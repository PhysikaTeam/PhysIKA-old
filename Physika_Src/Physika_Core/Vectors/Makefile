# Makefile for Vectors
# This makefile is by no means good! It needs to be replaced when I get better at Makefile.
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

# include top level makefile header
include $(PHYSIKA_ROOT_DIR)/Makefile_header

# headers in this library
VECTORS_HEADERS=vector_base.h vector_2d.h vector_3d.h vector.h vector_Nd.h

# additional headers that this library denpend on
ADDITIONAL_HEADERS=../Utilities/global_config.h ../Utilities/physika_assert.h

# target directory of the headers
TARGET_HEADERS_DIR=$(PHYSIKA_PUBLIC_DIR)/include/Physika_Core/Vectors

# copy header files to target include directory
define COPY_HEADERS
mkdir -p $(TARGET_HEADERS_DIR)
cp $(VECTORS_HEADERS) $(TARGET_HEADERS_DIR)
endef

# object files to be compiled for this library
VECTORS_OBJECTS=vector_2d.o vector_3d.o vector_Nd.o

# target name of this library
TARGET=libvectors.a


vectors: $(TARGET)

$(TARGET): $(VECTORS_OBJECTS)
	ar rs $@ $^; cp $(TARGET) $(PHYSIKA_PUBLIC_DIR)/lib/

$(VECTORS_OBJECTS): %.o: %.cpp $(VECTORS_HEADERS) $(ADDITIONAL_HEADERS) copy_headers
	$(CXX) $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -c $< -o $@

copy_headers:
	$(COPY_HEADERS);

.PHONY: cleanvectors
cleanvectors:
	$(RM) $(TARGET) $(VECTORS_OBJECTS) $(PHYSIKA_PUBLIC_DIR)/lib/$(TARGET)
	-rm -r $(TARGET_HEADERS_DIR)
