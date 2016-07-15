ANALYZE_DIR = /home/jesse/FemtoAnalysis/ProcessData/Analyze
UTILITIES_DIR = /home/jesse/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -Wall -I${ANALYZE_DIR} -I${UTILITIES_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, FitParameter.o FitChi2Histograms.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o LednickyFitter.o CfLite.o CfHeavy.o Types.o Faddeeva.o)


all: runLamKchP runLamKchPTest runLamKchPTest2 runLamKchPTest3 runCompareKchFix2 runFits runTest

runLamKchP: ${MY_LIBS} BuildLamKchP.C
	g++ BuildLamKchP.C ${MY_LIBS} -o runLamKchP $(LIBS) $(CFLAGS)

runLamKchPTest: ${MY_LIBS} BuildLamKchPTest.C
	g++ BuildLamKchPTest.C ${MY_LIBS} -o runLamKchPTest $(LIBS) $(CFLAGS)

runLamKchPTest2: ${MY_LIBS} BuildLamKchPTest2.C
	g++ BuildLamKchPTest2.C ${MY_LIBS} -o runLamKchPTest2 $(LIBS) $(CFLAGS)

runLamKchPTest3: ${MY_LIBS} BuildLamKchPTest3.C
	g++ BuildLamKchPTest3.C ${MY_LIBS} -o runLamKchPTest3 $(LIBS) $(CFLAGS)

runCompareKchFix2: ${MY_LIBS} CompareKchFix2.C
	g++ CompareKchFix2.C ${MY_LIBS} -o runCompareKchFix2 $(LIBS) $(CFLAGS)

runFits: ${MY_LIBS} BuildFits.C
	g++ BuildFits.C ${MY_LIBS} -o runFits $(LIBS) $(CFLAGS)

runTest: ${MY_LIBS} test.C
	g++ test.C ${MY_LIBS} -o runTest $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cc %.hh
	g++ -c $< $(CFLAGS) -o $@





clean:
	rm -f runLamKchP runLamKchPTest runLamKchPTest2 runLamKchPTest3 runCompareKchFix2 runFits runTest ${MY_LIBS}
