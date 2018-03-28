LIBS   = $(shell root-config --libs)
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

RM = rm

MY_LIBS = $(addprefix build/, ParticleCoor.o Types.o PIDMapping.o ThermCommon.o ThermParticle.o ThermV0Particle.o ThermEvent.o ThermChargedResidual.o ThermPairAnalysis.o ThermSingleParticleAnalysis.o ThermEventsCollection.o SimpleThermAnalysis.o)

BINARIES = runAnalysis

all: $(BINARIES)

runAnalysis: ${MY_LIBS} RunAnalysis.C
	g++ RunAnalysis.C ${MY_LIBS} -o runAnalysis $(LIBS) $(CFLAGS)

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@


clean:
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
