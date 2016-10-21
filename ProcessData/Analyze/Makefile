UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs)
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${UTILITIES_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, CfLite.o CfHeavy.o Cf2dLite.o Cf2dHeavy.o Types.o Purity.o PartialAnalysis.o Analysis.o DataAndModel.o PlotPartners.o PlotPartnersLamKch.o PlotPartnersLamK0.o CanvasPartition.o)

all: runBuildcLamcKch runBuildcLamK0 runBuildcXicKch runBuildcLamcLam runCombineConjugatesLamKch runCombineConjugatesLamK0 runCombineConjugatesXiKch runBuildMomResMatrix runBuildMomResMatrix2 runCompareFixes runCorrectCfs runTest

runBuildcLamcKch: ${MY_LIBS} BuildcLamcKchAnalyses.C
	g++ BuildcLamcKchAnalyses.C ${MY_LIBS} -o runBuildcLamcKch $(LIBS) $(CFLAGS)

runBuildcLamK0: ${MY_LIBS} BuildcLamK0Analyses.C
	g++ BuildcLamK0Analyses.C ${MY_LIBS} -o runBuildcLamK0 $(LIBS) $(CFLAGS)

runBuildcXicKch: ${MY_LIBS} BuildcXicKchAnalyses.C
	g++ BuildcXicKchAnalyses.C ${MY_LIBS} -o runBuildcXicKch $(LIBS) $(CFLAGS)

runBuildcLamcLam: ${MY_LIBS} BuildcLamcLamAnalyses.C
	g++ BuildcLamcLamAnalyses.C ${MY_LIBS} -o runBuildcLamcLam $(LIBS) $(CFLAGS)

runCombineConjugatesLamKch: ${MY_LIBS} CombineConjugatesLamKch.C
	g++ CombineConjugatesLamKch.C ${MY_LIBS} -o runCombineConjugatesLamKch $(LIBS) $(CFLAGS)

runCombineConjugatesLamK0: ${MY_LIBS} CombineConjugatesLamK0.C
	g++ CombineConjugatesLamK0.C ${MY_LIBS} -o runCombineConjugatesLamK0 $(LIBS) $(CFLAGS)

runCombineConjugatesXiKch: ${MY_LIBS} CombineConjugatesXiKch.C
	g++ CombineConjugatesXiKch.C ${MY_LIBS} -o runCombineConjugatesXiKch $(LIBS) $(CFLAGS)

runBuildMomResMatrix: ${MY_LIBS} BuildMomResMatrix.C
	g++ BuildMomResMatrix.C ${MY_LIBS} -o runBuildMomResMatrix $(LIBS) $(CFLAGS)

runBuildMomResMatrix2: ${MY_LIBS} BuildMomResMatrix2.C
	g++ BuildMomResMatrix2.C ${MY_LIBS} -o runBuildMomResMatrix2 $(LIBS) $(CFLAGS)

runCompareFixes: ${MY_LIBS} CompareFixes.C
	g++ CompareFixes.C ${MY_LIBS} -o runCompareFixes $(LIBS) $(CFLAGS)

runCorrectCfs: ${MY_LIBS} CorrectCfs.C
	g++ CorrectCfs.C ${MY_LIBS} -o runCorrectCfs $(LIBS) $(CFLAGS)

runTest: ${MY_LIBS} testPlotPartnersLamKch.C
	g++ testPlotPartnersLamKch.C ${MY_LIBS} -o runTest $(LIBS) $(CFLAGS)


build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cc %.hh
	g++ -c $< $(CFLAGS) -o $@





clean:
	rm -f runBuildcLamcKch runBuildcLamK0 runBuildcXicKch runBuildcLamcLam runCombineConjugatesLamKch runCombineConjugatesLamK0 runCombineConjugatesXiKch runBuildMomResMatrix runBuildMomResMatrix2 runCompareFixes runCorrectCfs runTest ${MY_LIBS}
