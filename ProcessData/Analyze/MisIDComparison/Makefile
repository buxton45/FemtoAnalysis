ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs)
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${UTILITIES_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, CfLite.o CfHeavy.o Cf2dLite.o Cf2dHeavy.o Types.o Purity.o PartialAnalysis.o Analysis.o DataAndModel.o PlotPartners.o PlotPartnersLamKch.o PlotPartnersLamK0.o SystematicAnalysis.o CanvasPartition.o)

all: runMassAssHyp

runMassAssHyp: ${MY_LIBS} BuildMassAssumingHypothesesComparisons.C
	g++ BuildMassAssumingHypothesesComparisons.C ${MY_LIBS} -o runMassAssHyp $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${ANALYZE_DIR}/%.cc ${ANALYZE_DIR}/%.hh
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@

clean:
	rm -f runMassAssHyp ${MY_LIBS}
