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
SYSTEMATICS_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/LednickyFitter/Systematics
COULOMBFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${LEDNICKYFITTER_DIR} -I${SYSTEMATICS_DIR} -I${COULOMBFITTER_DIR} -I${UTILITIES_DIR} -g -O0 -fopenmp
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, FitParameter.o FitChi2Histograms.o BackgroundFitter.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o MathematicaSession.o WaveFunction.o ChargedResidualCf.o SimpleChargedResidualCf.o NeutralResidualCf.o ResidualCollection.o LednickyFitter.o FitValuesWriter.o FitValuesLatexTableHelperWriter.o FitValuesWriterwSysErrs.o FitValuesLatexTableHelperWriterwSysErrs.o FitGenerator.o FitGeneratorAndDraw.o DualieFitGenerator.o ThermCf.o CfLite.o CfHeavy.o Types.o Types_LambdaValues.o Types_FitParamValues.o Types_ThermBgdParams.o Types_SysFileInfo.o Faddeeva.o FitSystematicAnalysis.o DualieFitSystematicAnalysis.o TripleFitGenerator.o TripleFitSystematicAnalysis.o CanvasPartition.o)

BINARIES = runDrawFinalResultswSysErrsTriple runmTScalingTriple runDrawTripleComparisons

all: $(BINARIES)

runDrawFinalResultswSysErrsTriple: ${MY_LIBS} DrawFinalResultswSysErrsTriple.C
	g++ DrawFinalResultswSysErrsTriple.C ${MY_LIBS} -I${INCDIR} -o runDrawFinalResultswSysErrsTriple -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runmTScalingTriple: ${MY_LIBS} DrawmTScalingTriple.C
	g++ DrawmTScalingTriple.C ${MY_LIBS} -I${INCDIR} -o runmTScalingTriple -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runDrawTripleComparisons: ${MY_LIBS} DrawTripleComparisons.C
	g++ DrawTripleComparisons.C ${MY_LIBS} -I${INCDIR} -o runDrawTripleComparisons -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${COULOMBFITTER_DIR}/%.cxx ${COULOMBFITTER_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cxx ${LEDNICKYFITTER_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cc ${LEDNICKYFITTER_DIR}/%.hh
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${SYSTEMATICS_DIR}/%.cxx ${SYSTEMATICS_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

clean:
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
