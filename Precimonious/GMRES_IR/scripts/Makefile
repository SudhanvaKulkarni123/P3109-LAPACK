CXXFLAGS = -std=c++20
CXX = g++

CDBG += -w
CXXFLAGS += $(CDBG) 
CXXFLAGS += -fopenmp 



SOURCES = GMRES_IR.cpp 
EXECUTABLE = GMRES_IR 
INCLUDE_PATH = -I$(HOME)/tlapack/include
INCLUDE_PATH += -I$(HOME)
INCLUDE_PATH += -I$(HOME)/P3109-LAPACK/Precimonious/utils



PYTHON_LIBS := $(shell python3.12-config --libs)
PYTHON_INCLUDE_DIRS := $(shell python3.12-config --includes)

PY_CFLAGS  := $(shell python3.12-config --cflags)
PY_LDFLAGS := $(shell python3.12-config --ldflags)


GMRES:
	$(CXX) -o GMRES_IR GMRES_IR.cpp  $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.12

LU :
	$(CXX) -o example_lu example_lu.cpp $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.12

MORRISON:
	$(CXX) -o example_morrison example_morrison.cpp $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.12

CG : 
	$(CXX) -o CG_IR CG_IR.cpp -v $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.12

FGMRES : 
	$(CXX) -o FGMRES_IR FGMRES_IR.cpp $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.12


$(EXECUTABLE): $(SOURCES)
	$(CXX) -o $@ $^  $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.12

CARSON :
	$(CXX) -o carson_counter carson_counter_eg.cpp $(CXXFLAGS) $(INCLUDE_PATH) $(PYTHON_INCLUDE_DIRS) $(PYTHON_LIBS) $(PY_CFLAGS) $(PY_LDFLAGS) -lpython3.12
clean:
	rm -f $(EXECUTABLE) *.txt *.out

	
all: $(SOURCES) $(EXECUTABLE)

