CXXFLAGS = -std=c++17
CXX = clang++-18
SOURCES = funarc_8.cpp
EXECUTABLE = funarc_8

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(CXX) -o $@ $^ $(CFLAGS)

clean:
	rm -f $(EXECUTABLE) *.txt *.out