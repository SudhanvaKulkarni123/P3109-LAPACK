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


GMRES:
	$(CXX) -o GMRES_IR GMRES_IR.cpp $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.8

LU :
	$(CXX) -o example_lu example_lu.cpp $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.8

MORRISON:
	$(CXX) -o example_morrison example_morrison.cpp $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.8


$(EXECUTABLE): $(SOURCES)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.8 

clean:
	rm -f $(EXECUTABLE) *.txt *.out

	
all: $(SOURCES) $(EXECUTABLE)

