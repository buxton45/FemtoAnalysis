ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${UTILITIES_DIR} -g -O0 -fopenmp
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, FitParameter.o FitChi2Histograms.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o LednickyFitter.o FitGenerator.o CfLite.o CfHeavy.o Types.o Faddeeva.o FitSystematicAnalysis.o CanvasPartition.o)


all: runLamKchP runLamKchPTest runCompareKchFix2 runFits runTest runGenerator runScattParamValues

runLamKchP: ${MY_LIBS} BuildLamKchP.C
	g++ BuildLamKchP.C ${MY_LIBS} -o runLamKchP $(LIBS) $(CFLAGS)

runLamKchPTest: ${MY_LIBS} BuildLamKchPTest.C
	g++ BuildLamKchPTest.C ${MY_LIBS} -o runLamKchPTest $(LIBS) $(CFLAGS)

runCompareKchFix2: ${MY_LIBS} CompareKchFix2.C
	g++ CompareKchFix2.C ${MY_LIBS} -o runCompareKchFix2 $(LIBS) $(CFLAGS)

runFits: ${MY_LIBS} BuildFits.C
	g++ BuildFits.C ${MY_LIBS} -o runFits $(LIBS) $(CFLAGS)

runTest: ${MY_LIBS} test.C
	g++ test.C ${MY_LIBS} -o runTest $(LIBS) $(CFLAGS)

runGenerator: ${MY_LIBS} testGenerator.C
	g++ testGenerator.C ${MY_LIBS} -o runGenerator $(LIBS) $(CFLAGS)

runScattParamValues: ${MY_LIBS} ScattParamValues.C
	g++ ScattParamValues.C ${MY_LIBS} -o runScattParamValues $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cc %.hh
	g++ -c $< $(CFLAGS) -o $@





clean:
	rm -f runLamKchP runLamKchPTest runCompareKchFix2 runFits runTest runGenerator runScattParamValues ${MY_LIBS}
