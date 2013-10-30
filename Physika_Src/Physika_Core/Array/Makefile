# Makefile for Array
# Simply copy the headerfiles to target include directory
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

# target directory of the headers
TARGET_HEADERS_DIR=$(PHYSIKA_PUBLIC_DIR)/include/Physika_Core/Array

# copy header files to target include directory
define COPY_HEADERS
mkdir -p $(TARGET_HEADERS_DIR)
cp *.h $(TARGET_HEADERS_DIR)
endef

utilities: 
	$(COPY_HEADERS)

.PHONY: cleanarray
cleanutilities:
	-rm -r $(TARGET_HEADERS_DIR)