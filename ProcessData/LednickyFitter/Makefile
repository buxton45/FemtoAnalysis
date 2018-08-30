VERSION=11.0
WSTPLINKDIR = /usr/local/Wolfram/Mathematica/10.4/SystemFiles/Links/WSTP/DeveloperKit
SYS = Linux-x86-64
CADDSDIR = ${WSTPLINKDIR}/${SYS}/CompilerAdditions

INCDIR = ${CADDSDIR}
LIBDIR = ${CADDSDIR}

EXTRALIBS = -lm -lpthread -lrt -lstdc++ -ldl -luuid
WSTPLIB = WSTP64i4

WSPREP = ${CADDSDIR}/wsprep
RM = rm
######################################################

ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
LEDNICKYFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/LednickyFitter
COULOMBFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${LEDNICKYFITTER_DIR} -I${COULOMBFITTER_DIR} -I${UTILITIES_DIR} -g -O0 -fopenmp
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, FitParameter.o FitChi2Histograms.o BackgroundFitter.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o MathematicaSession.o WaveFunction.o ChargedResidualCf.o SimpleChargedResidualCf.o NeutralResidualCf.o ResidualCollection.o SimpleLednickyFitter.o LednickyFitter.o FitValuesWriter.o FitValuesLatexTableHelperWriter.o FitValuesWriterwSysErrs.o FitValuesLatexTableHelperWriterwSysErrs.o FitGenerator.o FitGeneratorAndDraw.o DualieFitGenerator.o ThermCf.o CfLite.o CfHeavy.o Types.o Types_LambdaValues.o Types_FitParamValues.o Types_ThermBgdParams.o AnalysisInfo.o Faddeeva.o CanvasPartition.o)

BINARIES = runLamKchP runLamKchPTest runCompareKchFix2 runFits runTest runGenerator runDualieGenerator runCompareTwoAnalyses runCompareThreeAnalyses runCompareMixedcLamcKch runScattParamValues runLamKchPvsKchM runPlotAll runSimpleLednicky runCompareFittingMethods runCompareFittingMethodsv2 runmTScaling runCompareBackgrounds runCpaFit

all: $(BINARIES)

runLamKchP: ${MY_LIBS} BuildLamKchP.C
	g++ BuildLamKchP.C ${MY_LIBS} -I${INCDIR} -o runLamKchP -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runLamKchPTest: ${MY_LIBS} BuildLamKchPTest.C
	g++ BuildLamKchPTest.C ${MY_LIBS} -I${INCDIR} -o runLamKchPTest -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runCompareKchFix2: ${MY_LIBS} CompareKchFix2.C
	g++ CompareKchFix2.C ${MY_LIBS} -I${INCDIR} -o runCompareKchFix2 -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runFits: ${MY_LIBS} BuildFits.C
	g++ BuildFits.C ${MY_LIBS} -I${INCDIR} -o runFits -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runTest: ${MY_LIBS} test.C
	g++ test.C ${MY_LIBS} -I${INCDIR} -o runTest -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runGenerator: ${MY_LIBS} GenerateFits.C
	g++ GenerateFits.C ${MY_LIBS} -I${INCDIR} -o runGenerator -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runDualieGenerator: ${MY_LIBS} DualieGenerateFits.C
	g++ DualieGenerateFits.C ${MY_LIBS} -I${INCDIR} -o runDualieGenerator -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runCompareTwoAnalyses: ${MY_LIBS} CompareTwoAnalyses.C
	g++ CompareTwoAnalyses.C ${MY_LIBS} -I${INCDIR} -o runCompareTwoAnalyses -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runCompareThreeAnalyses: ${MY_LIBS} CompareThreeAnalyses.C
	g++ CompareThreeAnalyses.C ${MY_LIBS} -I${INCDIR} -o runCompareThreeAnalyses -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runCompareMixedcLamcKch: ${MY_LIBS} CompareMixedcLamcKch.C
	g++ CompareMixedcLamcKch.C ${MY_LIBS} -I${INCDIR} -o runCompareMixedcLamcKch -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runScattParamValues: ${MY_LIBS} ScattParamValues.C
	g++ ScattParamValues.C ${MY_LIBS} -I${INCDIR} -o runScattParamValues -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runLamKchPvsKchM: ${MY_LIBS} DrawLamKchPvsLamKchM.C
	g++ DrawLamKchPvsLamKchM.C ${MY_LIBS} -I${INCDIR} -o runLamKchPvsKchM -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runPlotAll: ${MY_LIBS} PlotAllFitsCentral.C
	g++ PlotAllFitsCentral.C ${MY_LIBS} -I${INCDIR} -o runPlotAll -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runSimpleLednicky: ${MY_LIBS} RunSimpleLednickyFitter.C
	g++ RunSimpleLednickyFitter.C ${MY_LIBS} -I${INCDIR} -o runSimpleLednicky -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runCompareFittingMethods: ${MY_LIBS} CompareFittingMethods.C
	g++ CompareFittingMethods.C ${MY_LIBS} -I${INCDIR} -o runCompareFittingMethods -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runCompareFittingMethodsv2: ${MY_LIBS} CompareFittingMethodsv2.C
	g++ CompareFittingMethodsv2.C ${MY_LIBS} -I${INCDIR} -o runCompareFittingMethodsv2 -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runmTScaling: ${MY_LIBS} DrawmTScaling.C
	g++ DrawmTScaling.C ${MY_LIBS} -I${INCDIR} -o runmTScaling -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runCompareBackgrounds: ${MY_LIBS} CompareBackgrounds.C
	g++ CompareBackgrounds.C ${MY_LIBS} -I${INCDIR} -o runCompareBackgrounds -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runCpaFit: ${MY_LIBS} CpaTemplateFit.C
	g++ CpaTemplateFit.C ${MY_LIBS} -I${INCDIR} -o runCpaFit -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${COULOMBFITTER_DIR}/%.cxx ${COULOMBFITTER_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: %.cc %.hh
	g++ -c $< $(CFLAGS) -o $@





clean:
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
