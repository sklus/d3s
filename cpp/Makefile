CC = g++

ENVROOT = $(CONDA_PREFIX)

ifeq ($(ENVROOT),)
$(error - You must activate Conda environment)
endif

INCDIR  = $(ENVROOT)/include
LIBDIR = $(ENVROOT)/lib
CFLAGS = -c -O3 -fPIC -D_UNIX -std=c++11 -Wno-deprecated-declarations -I$(INCDIR) -I$(INCDIR)/boost -I$(INCDIR)/python3.5m/

LFLAGS = -O3 -shared $(LIBDIR)/libboost_python35.so $(LIBDIR)/libboost_system.so $(LIBDIR)/libpython3.5m.so $(LIBDIR)/libboost_numpy35.so -Wl,-rpath=$(ENVROOT)/lib

systems: systems.cpp
	$(CC) $(CFLAGS) $^
	$(CC) -Wl,-soname,$@.so -Wl,-rpath=$(LIBDIR) $@.o -o ../d3s/$@.so $(LFLAGS)

.PHONY: clean

clean:
	rm -f *.o
