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

MY_LIBS = $(addprefix build/, FitParameter.o FitChi2Histograms.o BackgroundFitter.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o MathematicaSession.o WaveFunction.o ChargedResidualCf.o SimpleChargedResidualCf.o NeutralResidualCf.o ResidualCollection.o LednickyFitter.o FitGenerator.o CfLite.o CfHeavy.o Types.o Types_LambdaValues.o Types_FitParamValues.o Types_SysFileInfo.o Faddeeva.o FitSystematicAnalysis.o CanvasPartition.o)

all: runSingleSystematics runAllSystematics runRead runFindOutliers runFitRange runNonFlatBgdSys

runSingleSystematics: ${MY_LIBS} BuildSingleCutSystematics.C
	g++ BuildSingleCutSystematics.C ${MY_LIBS} -I${INCDIR} -o runSingleSystematics -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runAllSystematics: ${MY_LIBS} BuildAllCutsSystematics.C
	g++ BuildAllCutsSystematics.C ${MY_LIBS} -I${INCDIR} -o runAllSystematics -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runRead: ${MY_LIBS} ReadSystematicFitResults.C
	g++ ReadSystematicFitResults.C ${MY_LIBS} -I${INCDIR} -o runRead -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runFindOutliers: ${MY_LIBS} FindOutliers.C
	g++ FindOutliers.C ${MY_LIBS} -I${INCDIR} -o runFindOutliers -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runFitRange: ${MY_LIBS} FitRangeSystematics.C
	g++ FitRangeSystematics.C ${MY_LIBS} -I${INCDIR} -o runFitRange -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runNonFlatBgdSys: ${MY_LIBS} NonFlatBgdFitTypeSystematics.C
	g++ NonFlatBgdFitTypeSystematics.C ${MY_LIBS} -I${INCDIR} -o runNonFlatBgdSys -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${COULOMBFITTER_DIR}/%.cxx ${COULOMBFITTER_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cxx ${LEDNICKYFITTER_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cc ${LEDNICKYFITTER_DIR}/%.hh
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

clean:
	rm -f runSingleSystematics runAllSystematics runRead runFindOutliers runFitRange runNonFlatBgdSys ${MY_LIBS}
