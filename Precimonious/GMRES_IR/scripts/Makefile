CXXFLAGS = -std=c++20  
CXX = clang++

CDBG += -w
CXXFLAGS += $(CDBG)


SOURCES = GMRES_IR.cpp 
EXECUTABLE = GMRES_IR 
INCLUDE_PATH = -I$(HOME)/Documents/Precimonious-Clang-Plugins/TLAPACK-Precimonious-exp/Precimonious/tlapack/include
INCLUDE_PATH += -I$(HOME)/Documents/Precimonious-Clang-Plugins/TLAPACK-Precimonious-exp/Precimonious/utilites



PYTHON_LIBS := $(shell python3-config --libs)
PYTHON_INCLUDE_DIRS := $(shell python3-config --includes)

PY_CFLAGS  := $(shell python3-config --cflags)
PY_LDFLAGS := $(shell python3-config --ldflags)


GMRES:
	$(CXX) -o GMRES_IR GMRES_IR.cpp  $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.11

LU :
	$(CXX) -o example_lu example_lu.cpp $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.11

MORRISON:
	$(CXX) -o example_morrison example_morrison.cpp $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.11

CG : 
	$(CXX) -o CG_IR CG_IR.cpp $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.11


$(EXECUTABLE): $(SOURCES)
	$(CXX) -o $@ $^  $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.11

CARSON :
	$(CXX) -o carson_counter carson_counter_eg.cpp $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.11

clean:
	rm -f $(EXECUTABLE) *.txt *.out

	
all: $(SOURCES) $(EXECUTABLE)

