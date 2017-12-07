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

MY_LIBS = $(addprefix build/, FitParameter.o FitChi2Histograms.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o MathematicaSession.o WaveFunction.o ChargedResidualCf.o SimpleChargedResidualCf.o NeutralResidualCf.o ResidualCollection.o SimpleLednickyFitter.o LednickyFitter.o FitGenerator.o CfLite.o CfHeavy.o Types.o Types_LambdaValues.o Types_FitParamValues.o AnalysisInfo.o Faddeeva.o FitSystematicAnalysis.o CanvasPartition.o)

BINARIES = runLamKchP runLamKchPTest runCompareKchFix2 runFits runTest runGenerator runCompareIgnoreOnFlyStatus runScattParamValues runLamKchPvsKchM runPlotAll runSimpleLednicky runCompareFittingMethods runmTScaling

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

runCompareIgnoreOnFlyStatus: ${MY_LIBS} CompareIgnoreOnFlyStatus.C
	g++ CompareIgnoreOnFlyStatus.C ${MY_LIBS} -I${INCDIR} -o runCompareIgnoreOnFlyStatus -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

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

runmTScaling: ${MY_LIBS} DrawmTScaling.C
	g++ DrawmTScaling.C ${MY_LIBS} -I${INCDIR} -o runmTScaling -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

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
