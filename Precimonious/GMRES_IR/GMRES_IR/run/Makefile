CXXFLAGS = -std=c++20  -I/usr/include/python3.8
CXX = clang++-19

CDBG += -w
CXXFLAGS += $(CDBG)

SOURCES = GMRES_IR.cpp 
EXECUTABLE = GMRES_IR 
INCLUDE_PATH = -I/root/home/Precimonious/tlapack/include
GCC_PATH = /usr/lib/


PYTHON_LIBS := $(shell python3-config --libs)
PYTHON_INCLUDE_DIRS := $(shell python3-config --includes)

PY_CFLAGS  := $(shell python3-config --cflags)
PY_LDFLAGS := $(shell python3-config --ldflags)


create-space:
	$(CXX) -Xclang -load -Xclang ../../plugin/CreateSearchSpace.so -Xclang -plugin -Xclang create-space -Xclang -plugin-arg-create-space -Xclang -output-path -Xclang -plugin-arg-create-space -Xclang ./ -Xclang -plugin-arg-create-space -Xclang -output-name -Xclang -plugin-arg-create-space -Xclang config.json -Xclang -plugin-arg-create-space -Xclang -input-file -Xclang -plugin-arg-create-space -Xclang GMRES_IR ../scripts/GMRES_IR $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.8

	


clean:
	rm -f *.out  *.txt  *output
