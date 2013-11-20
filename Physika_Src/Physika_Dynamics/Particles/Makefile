# Makefile for Particles
# This makefile is by no means good! It needs to be replaced when I get better at Makefile.
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

# include top level makefile header
include $(PHYSIKA_ROOT_DIR)/Makefile_header

# headers in this library
PARTICLES_HEADERS=particle.h

# additional headers that this library denpend on
ADDITIONAL_HEADERS=

# target directory of the headers
TARGET_HEADERS_DIR=$(PHYSIKA_PUBLIC_DIR)/include/Physika_Dynamics/Particles

# copy header files to target include directory
define COPY_HEADERS
mkdir -p $(TARGET_HEADERS_DIR)
cp $(PARTICLES_HEADERS) $(TARGET_HEADERS_DIR)
endef

# object files to be compiled for this library
PARTICLES_OBJECTS=particle.o

# target name of this library
TARGET=libparticles.a


particles: $(TARGET)

$(TARGET): $(PARTICLES_OBJECTS)
	ar rs $@ $^; cp $(TARGET) $(PHYSIKA_PUBLIC_DIR)/lib/

$(PARTICLES_OBJECTS): %.o: %.cpp $(PARTICLES_HEADERS) $(ADDITIONAL_HEADERS) copy_headers
	$(CXX) $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -c $< -o $@

copy_headers:
	$(COPY_HEADERS);

.PHONY: cleanparticles
cleanparticles:
	$(RM) $(TARGET) $(PARTICLES_OBJECTS) $(PHYSIKA_PUBLIC_DIR)/lib/$(TARGET)
	-rm -r $(TARGET_HEADERS_DIR)
