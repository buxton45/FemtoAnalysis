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

MY_LIBS = $(addprefix build/, FitParameter.o FitChi2Histograms.o BackgroundFitter.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o MathematicaSession.o WaveFunction.o ChargedResidualCf.o SimpleChargedResidualCf.o NeutralResidualCf.o ResidualCollection.o LednickyFitter.o FitValuesWriter.o FitValuesLatexTableHelperWriter.o FitValuesWriterwSysErrs.o FitValuesLatexTableHelperWriterwSysErrs.o FitGenerator.o FitGeneratorAndDraw.o DualieFitGenerator.o ThermCf.o CfLite.o CfHeavy.o Types.o Types_LambdaValues.o Types_FitParamValues.o Types_ThermBgdParams.o Types_SysFileInfo.o Faddeeva.o FitSystematicAnalysis.o DualieFitSystematicAnalysis.o CanvasPartition.o)

BINARIES = runSingleSystematics runAllSystematics runRead runFindOutliers runFitRange runNonFlatBgdSys runTest runBuildLatexTableHelperwSysErrs runBuildLatexTableHelperwSysErrsForSingle runCompareFittingMethodswSysErrs runCompareFittingMethodswSysErrsForAnNote runSingleDualieSystematics runAllDualieSystematics runFitRangeDualie runNonFlatBgdDualieSys

all: $(BINARIES)

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

runBuildLatexTableHelperwSysErrs: ${MY_LIBS} BuildLatexTableHelperwSysErrs.C
	g++ BuildLatexTableHelperwSysErrs.C ${MY_LIBS} -I${INCDIR} -o runBuildLatexTableHelperwSysErrs -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runBuildLatexTableHelperwSysErrsForSingle: ${MY_LIBS} BuildLatexTableHelperwSysErrsForSingle.C
	g++ BuildLatexTableHelperwSysErrsForSingle.C ${MY_LIBS} -I${INCDIR} -o runBuildLatexTableHelperwSysErrsForSingle -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runTest: ${MY_LIBS} test.C
	g++ test.C ${MY_LIBS} -I${INCDIR} -o runTest -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runCompareFittingMethodswSysErrs: ${MY_LIBS} CompareFittingMethodswSysErrs.C
	g++ CompareFittingMethodswSysErrs.C ${MY_LIBS} -I${INCDIR} -o runCompareFittingMethodswSysErrs -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runCompareFittingMethodswSysErrsForAnNote: ${MY_LIBS} CompareFittingMethodswSysErrsForAnNote.C
	g++ CompareFittingMethodswSysErrsForAnNote.C ${MY_LIBS} -I${INCDIR} -o runCompareFittingMethodswSysErrsForAnNote -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runSingleDualieSystematics: ${MY_LIBS} BuildSingleCutDualieSystematics.C
	g++ BuildSingleCutDualieSystematics.C ${MY_LIBS} -I${INCDIR} -o runSingleDualieSystematics -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runAllDualieSystematics: ${MY_LIBS} BuildAllCutsDualieSystematics.C
	g++ BuildAllCutsDualieSystematics.C ${MY_LIBS} -I${INCDIR} -o runAllDualieSystematics -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runFitRangeDualie: ${MY_LIBS} FitRangeDualieSystematics.C
	g++ FitRangeDualieSystematics.C ${MY_LIBS} -I${INCDIR} -o runFitRangeDualie -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runNonFlatBgdDualieSys: ${MY_LIBS} NonFlatBgdFitTypeDualieSystematics.C
	g++ NonFlatBgdFitTypeDualieSystematics.C ${MY_LIBS} -I${INCDIR} -o runNonFlatBgdDualieSys -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

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
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
