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
THERMINATOR_DIR = /home/jesse/Analysis/FemtoAnalysis/Therminator
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

##gsl-config stuff needed to use Adam's SpherHarmonics and CorrFctnDirectYlm
LIBS   = $(shell gsl-config --libs) $(shell root-config --libs) -lMinuit
CFLAGS = $(shell gsl-config --cflags) $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${LEDNICKYFITTER_DIR} -I${SYSTEMATICS_DIR} -I${COULOMBFITTER_DIR} -I${THERMINATOR_DIR} -I${UTILITIES_DIR} -g -O0 -fopenmp
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, FitParameter.o FitChi2Histograms.o BackgroundFitter.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o MathematicaSession.o WaveFunction.o ChargedResidualCf.o SimpleChargedResidualCf.o NeutralResidualCf.o ResidualCollection.o LednickyFitter.o FitValuesWriter.o FitValuesLatexTableHelperWriter.o FitValuesWriterwSysErrs.o FitValuesLatexTableHelperWriterwSysErrs.o FitGenerator.o FitGeneratorAndDraw.o DualieFitGenerator.o ThermCf.o CfLite.o CfHeavy.o Types.o Types_LambdaValues.o Types_FitParamValues.o Types_ThermBgdParams.o Types_SysFileInfo.o Faddeeva.o FitSystematicAnalysis.o DualieFitSystematicAnalysis.o TripleFitGenerator.o TripleFitSystematicAnalysis.o CanvasPartition.o Cf2dLite.o Cf2dHeavy.o AnalysisInfo.o Purity.o PartialAnalysis.o Analysis.o DataAndModel.o SystematicAnalysis.o SpherHarmonics.o CorrFctnDirectYlm.o CorrFctnDirectYlmLite.o CorrFctnDirectYlmHeavy.o CorrFctnDirectYlmTherm.o SuperpositionFitBgd.o PIDMapping.o NumIntLednickyCf.o HistInfoPrinter.o)

BINARIES = runDrawFinalResultswSysErrsTriple runmTScalingTriple runmTScalingTriplev2 runmTScalingTriplev3 runComparisonsTriple runPurity runBuildMomResMatrix runThermBgds runThermBgdsCompFlow runThermFlow runAvgSepXiKch runAvgSepLamK runStavMethComp runGetTotCollSizes runCompLamK runPlotAllFitsCentral runPlotAllFitsTriple runFitThermCfs

all: $(BINARIES)

runDrawFinalResultswSysErrsTriple: ${MY_LIBS} DrawFinalResultswSysErrsTriple.C
	g++ DrawFinalResultswSysErrsTriple.C ${MY_LIBS} -I${INCDIR} -o runDrawFinalResultswSysErrsTriple -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runmTScalingTriple: ${MY_LIBS} DrawmTScalingTriple.C
	g++ DrawmTScalingTriple.C ${MY_LIBS} -I${INCDIR} -o runmTScalingTriple -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runmTScalingTriplev2: ${MY_LIBS} DrawmTScalingTriplev2.C
	g++ DrawmTScalingTriplev2.C ${MY_LIBS} -I${INCDIR} -o runmTScalingTriplev2 -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)
	
runmTScalingTriplev3: ${MY_LIBS} DrawmTScalingTriplev3.C
	g++ DrawmTScalingTriplev3.C ${MY_LIBS} -I${INCDIR} -o runmTScalingTriplev3 -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)
	

runComparisonsTriple: ${MY_LIBS} DrawComparisonsTriple.C
	g++ DrawComparisonsTriple.C ${MY_LIBS} -I${INCDIR} -o runComparisonsTriple -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runPurity: ${MY_LIBS} BuildPurityPlots.C
	g++ BuildPurityPlots.C ${MY_LIBS} -I${INCDIR} -o runPurity -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runBuildMomResMatrix: ${MY_LIBS} BuildMomResMatrix.C
	g++ BuildMomResMatrix.C ${MY_LIBS} -I${INCDIR} -o runBuildMomResMatrix -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runThermBgds: ${MY_LIBS} ThermBackgrounds.C
	g++ ThermBackgrounds.C ${MY_LIBS} -I${INCDIR} -o runThermBgds -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runThermBgdsCompFlow: ${MY_LIBS} ThermBgdsCompareFlow.C
	g++ ThermBgdsCompareFlow.C ${MY_LIBS} -I${INCDIR} -o runThermBgdsCompFlow -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runThermFlow: ${MY_LIBS} DrawThermFlow.C
	g++ DrawThermFlow.C ${MY_LIBS} -I${INCDIR} -o runThermFlow -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runAvgSepXiKch: ${MY_LIBS} AvgSepXiKch.C
	g++ AvgSepXiKch.C ${MY_LIBS} -I${INCDIR} -o runAvgSepXiKch -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runAvgSepLamK: ${MY_LIBS} AvgSepLamK.C
	g++ AvgSepLamK.C ${MY_LIBS} -I${INCDIR} -o runAvgSepLamK -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runStavMethComp: ${MY_LIBS} StavMethodComparisons.C
	g++ StavMethodComparisons.C ${MY_LIBS} -I${INCDIR} -o runStavMethComp -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runGetTotCollSizes: ${MY_LIBS} GetTotalCollectionSizes.C
	g++ GetTotalCollectionSizes.C ${MY_LIBS} -I${INCDIR} -o runGetTotCollSizes -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runCompLamK: ${MY_LIBS} CompareLamKchPKchMK0s.C
	g++ CompareLamKchPKchMK0s.C ${MY_LIBS} -I${INCDIR} -o runCompLamK -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runPlotAllFitsCentral: ${MY_LIBS} PlotAllFitsCentral.C
	g++ PlotAllFitsCentral.C ${MY_LIBS} -I${INCDIR} -o runPlotAllFitsCentral -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)
	
runPlotAllFitsTriple: ${MY_LIBS} PlotAllFitsTriple.C
	g++ PlotAllFitsTriple.C ${MY_LIBS} -I${INCDIR} -o runPlotAllFitsTriple -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

runFitThermCfs: ${MY_LIBS} FitThermCfs.C
	g++ FitThermCfs.C ${MY_LIBS} -I${INCDIR} -o runFitThermCfs -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS)

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

build/%.o: ${THERMINATOR_DIR}/%.cxx ${THERMINATOR_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

clean:
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
