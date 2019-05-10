CXX = mpic++ # or mpicxx
CXXFLAGS = -std=c++11 -O3 -fopenmp #-march=native

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
