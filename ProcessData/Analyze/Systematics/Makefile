
RM = rm

ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs)
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${UTILITIES_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, CfLite.o CfHeavy.o Cf2dLite.o Cf2dHeavy.o Types.o Types_SysFileInfo.o Purity.o PartialAnalysis.o Analysis.o DataAndModel.o PlotPartners.o PlotPartnersLamKch.o PlotPartnersLamK0.o SystematicAnalysis.o CanvasPartition.o)

BINARIES = runSingleSys runAllSysLamK0 runAllSysLamKch runAllSysXiKch runCompare runBuildErrorBars runBuildErrorBarsXiKch runBuildLeadingErrorContributors

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

runBuildErrorBarsXiKch: ${MY_LIBS} BuildErrorBarsXiKch.C
	g++ BuildErrorBarsXiKch.C ${MY_LIBS} -o runBuildErrorBarsXiKch $(LIBS) $(CFLAGS)

runBuildLeadingErrorContributors: ${MY_LIBS} BuildLeadingErrorContributors.C
	g++ BuildLeadingErrorContributors.C ${MY_LIBS} -o runBuildLeadingErrorContributors $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${ANALYZE_DIR}/%.cc ${ANALYZE_DIR}/%.hh
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@

clean:
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
