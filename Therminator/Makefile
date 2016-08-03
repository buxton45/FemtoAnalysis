ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${UTILITIES_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, Types.o ThermParticle.o ThermV0Particle.o ThermEvent.o ThermEventsCollection.o)


all: runTest

runTest: ${MY_LIBS} test.C
	g++ test.C ${MY_LIBS} -o runTest $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@


clean:
	rm -f runTest ${MY_LIBS}
