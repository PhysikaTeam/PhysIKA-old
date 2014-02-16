# Makefile for Utilities
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

# target directory of the headers
TARGET_HEADERS_DIR=$(PHYSIKA_PUBLIC_DIR)/include/Physika_Core/Utilities

# copy header files to target include directory
define COPY_HEADERS
mkdir -p $(TARGET_HEADERS_DIR)
cp *.h $(TARGET_HEADERS_DIR)
endef

utilities: 
	$(COPY_HEADERS)
	cd ./Timer; make timer

.PHONY: cleanutilities
cleanutilities:
	-rm -r $(TARGET_HEADERS_DIR)
	cd ./Timer; make cleantimer
