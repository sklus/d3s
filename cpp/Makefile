CC     = g++
BOOST  = /srv/public/sklus/boost_1_65
CFLAGS = -c -O3 -fPIC -D_UNIX -std=c++11 -Wno-deprecated-declarations -I$(BOOST)/include -I/storage/mi/sklus/miniconda3/include/python3.6m/
LFLAGS = -O3 -shared $(BOOST)/lib/libboost_python3.so $(BOOST)/lib/libboost_numpy3.so -L/storage/mi/sklus/miniconda3/lib/ -lpython3.6m -lm

systems: systems.cpp
	$(CC) $(CFLAGS) $^
	$(CC) $(LFLAGS) -Wl,-soname,$@.so -Wl,-rpath=$(BOOST)/lib -Wl,-rpath=/storage/mi/sklus/miniconda3/lib/ $@.o -o ../d3s/$@.so

.PHONY: clean

clean:
	rm -f *.o
