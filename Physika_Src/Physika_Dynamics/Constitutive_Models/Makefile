# Makefile for Constitutive Models
# This makefile is by no means good! It needs to be replaced when I get better at Makefile.
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

# include top level makefile header
include $(PHYSIKA_ROOT_DIR)/Makefile_header

# headers in this library
CONSTITUTIVE_MODELS_HEADERS=constitutive_model.h isotropic_hyperelastic_material.h neo_hookean.h st_venant_kirchhoff.h

# additional headers that this library denpend on
ADDITIONAL_HEADERS=

# target directory of the headers
TARGET_HEADERS_DIR=$(PHYSIKA_PUBLIC_DIR)/include/Physika_Dynamics/Constitutive_Models

# copy header files to target include directory
define COPY_HEADERS
mkdir -p $(TARGET_HEADERS_DIR)
cp $(CONSTITUTIVE_MODELS_HEADERS) $(TARGET_HEADERS_DIR)
endef

# object files to be compiled for this library
CONSTITUTIVE_MODELS_OBJECTS=neo_hookean.o st_venant_kirchhoff.o

# target name of this library
TARGET=libconstitutive_models.a


constitutive_models: $(TARGET)

$(TARGET): $(CONSTITUTIVE_MODELS_OBJECTS)
	ar rs $@ $^; cp $(TARGET) $(PHYSIKA_PUBLIC_DIR)/lib/

$(CONSTITUTIVE_MODELS_OBJECTS): %.o: %.cpp $(CONSTITUTIVE_MODELS_HEADERS) $(ADDITIONAL_HEADERS) copy_headers
	$(CXX) $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -c $< -o $@

copy_headers:
	$(COPY_HEADERS);

.PHONY: cleanconstitutivemodels
cleanconstitutivemodels:
	$(RM) $(TARGET) $(CONSTITUTIVE_MODELS_OBJECTS) $(PHYSIKA_PUBLIC_DIR)/lib/$(TARGET)
	-rm -r $(TARGET_HEADERS_DIR)
