ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
LEDNICKYFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/LednickyFitter
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${LEDNICKYFITTER_DIR} -I${UTILITIES_DIR} -g -O0 -fopenmp
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, FitParameter.o FitChi2Histograms.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o LednickyFitter.o FitGenerator.o CfLite.o CfHeavy.o Types.o Faddeeva.o FitSystematicAnalysis.o CanvasPartition.o)

all: runSingleSystematics runAllSystematics runRead runFitRange

runSingleSystematics: ${MY_LIBS} BuildSingleCutSystematics.C
	g++ BuildSingleCutSystematics.C ${MY_LIBS} -o runSingleSystematics $(LIBS) $(CFLAGS)

runAllSystematics: ${MY_LIBS} BuildAllCutsSystematics.C
	g++ BuildAllCutsSystematics.C ${MY_LIBS} -o runAllSystematics $(LIBS) $(CFLAGS)

runRead: ${MY_LIBS} ReadSystematicFitResults.C
	g++ ReadSystematicFitResults.C ${MY_LIBS} -o runRead $(LIBS) $(CFLAGS)

runFitRange: ${MY_LIBS} FitRangeSystematics.C
	g++ FitRangeSystematics.C ${MY_LIBS} -o runFitRange $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cxx ${LEDNICKYFITTER_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cc ${LEDNICKYFITTER_DIR}/%.hh
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@

clean:
	rm -f runSingleSystematics runAllSystematics runRead runFitRange ${MY_LIBS}
