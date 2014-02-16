# Makefile for Timer
# Author: Fei Zhu

PHYSIKA_ROOT_DIR=../../../..
PHYSIKA_PUBLIC_DIR=$(PHYSIKA_ROOT_DIR)/Public_Library

# include top level makefile header
include $(PHYSIKA_ROOT_DIR)/Makefile_header

# headers
TIMER_HEADER=timer.h

# target directory of the headers
TARGET_HEADERS_DIR=$(PHYSIKA_PUBLIC_DIR)/include/Physika_Core/Utilities/Timer

# copy header files to target include directory
define COPY_HEADERS
mkdir -p $(TARGET_HEADERS_DIR)
cp $(TIMER_HEADER) $(TARGET_HEADERS_DIR)
endef

# object files to be compiled
TIMER_OBJECT=timer.o

# target name
TARGET=libtimer.a

timer: $(TARGET)
	

$(TARGET): $(TIMER_OBJECT)
	ar rs $@ $^; cp $(TARGET) $(PHYSIKA_PUBLIC_DIR)/lib/

$(TIMER_OBJECT): %.o: %.cpp $(TIMER_HEADER) copy_headers
	$(CXX) $(CXXFLAGS) -I $(PHYSIKA_PUBLIC_DIR)/include -c $< -o $@

copy_headers:
	$(COPY_HEADERS)

.PHONY: cleantimer
cleantimer:
	$(RM) $(TARGET) $(TIMER_OBJECT) $(PHYSIKA_PUBLIC_DIR)/lib/$(TARGET)
