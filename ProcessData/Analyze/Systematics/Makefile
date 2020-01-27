
RM = rm

ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

##gsl-config stuff needed to use Adam's SpherHarmonics and CorrFctnDirectYlm
LIBS   = $(shell gsl-config --libs) $(shell root-config --libs)
CFLAGS = $(shell gsl-config --cflags) $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${UTILITIES_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, CfLite.o CfHeavy.o Cf2dLite.o Cf2dHeavy.o Types.o Types_SysFileInfo.o Purity.o PartialAnalysis.o Analysis.o DataAndModel.o PlotPartners.o PlotPartnersLamKch.o PlotPartnersLamK0.o SystematicAnalysis.o CanvasPartition.o SpherHarmonics.o CorrFctnDirectYlm.o CorrFctnDirectYlmLite.o CorrFctnDirectYlmHeavy.o CorrFctnDirectYlmTherm.o ThermCf.o)

BINARIES = runSingleSys runAllSysLamK0 runAllSysLamKch runAllSysXiKch runCompare runBuildErrorBars runBuildErrorBarsYlm runBuildErrorBarsXiKch runBuildLeadingErrorContributors runBuildRelativeContributions

all: $(BINARIES)

runSingleSys: ${MY_LIBS} BuildSingleSystematics.C
	g++ BuildSingleSystematics.C ${MY_LIBS} -o runSingleSys $(LIBS) $(CFLAGS)

runAllSysLamK0: ${MY_LIBS} BuildAllSystematicsLamK0.C
	g++ BuildAllSystematicsLamK0.C ${MY_LIBS} -o runAllSysLamK0 $(LIBS) $(CFLAGS)

runAllSysLamKch: ${MY_LIBS} BuildAllSystematicsLamKch.C
	g++ BuildAllSystematicsLamKch.C ${MY_LIBS} -o runAllSysLamKch $(LIBS) $(CFLAGS)

runAllSysXiKch: ${MY_LIBS} BuildAllSystematicsXiKch.C
	g++ BuildAllSystematicsXiKch.C ${MY_LIBS} -o runAllSysXiKch $(LIBS) $(CFLAGS)

runCompare: ${MY_LIBS} CompareDiffFitResults.C
	g++ CompareDiffFitResults.C ${MY_LIBS} -o runCompare $(LIBS) $(CFLAGS)

runBuildErrorBars: ${MY_LIBS} BuildErrorBars.C
	g++ BuildErrorBars.C ${MY_LIBS} -o runBuildErrorBars $(LIBS) $(CFLAGS)

runBuildErrorBarsYlm: ${MY_LIBS} BuildErrorBarsYlm.C
	g++ BuildErrorBarsYlm.C ${MY_LIBS} -o runBuildErrorBarsYlm $(LIBS) $(CFLAGS)

runBuildErrorBarsXiKch: ${MY_LIBS} BuildErrorBarsXiKch.C
	g++ BuildErrorBarsXiKch.C ${MY_LIBS} -o runBuildErrorBarsXiKch $(LIBS) $(CFLAGS)

runBuildLeadingErrorContributors: ${MY_LIBS} BuildLeadingErrorContributors.C
	g++ BuildLeadingErrorContributors.C ${MY_LIBS} -o runBuildLeadingErrorContributors $(LIBS) $(CFLAGS)
	
runBuildRelativeContributions: ${MY_LIBS} BuildRelativeContributions.C
	g++ BuildRelativeContributions.C ${MY_LIBS} -o runBuildRelativeContributions $(LIBS) $(CFLAGS)	

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${ANALYZE_DIR}/%.cc ${ANALYZE_DIR}/%.hh
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@

clean:
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
