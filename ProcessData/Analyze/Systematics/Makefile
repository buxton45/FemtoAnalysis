ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs)
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${UTILITIES_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, CfLite.o CfHeavy.o Cf2dLite.o Cf2dHeavy.o Types.o Purity.o PartialAnalysis.o Analysis.o DataAndModel.o PlotPartners.o PlotPartnersLamKch.o PlotPartnersLamK0.o SystematicAnalysis.o CanvasPartition.o)

all: runSingleSys runAllSysLamK0 runAllSysLamKch

runSingleSys: ${MY_LIBS} BuildSingleSystematics.C
	g++ BuildSingleSystematics.C ${MY_LIBS} -o runSingleSys $(LIBS) $(CFLAGS)

runAllSysLamK0: ${MY_LIBS} BuildAllSystematicsLamK0.C
	g++ BuildAllSystematicsLamK0.C ${MY_LIBS} -o runAllSysLamK0 $(LIBS) $(CFLAGS)

runAllSysLamKch: ${MY_LIBS} BuildAllSystematicsLamKch.C
	g++ BuildAllSystematicsLamKch.C ${MY_LIBS} -o runAllSysLamKch $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${ANALYZE_DIR}/%.cc ${ANALYZE_DIR}/%.hh
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@

clean:
	rm -f runSingleSys runAllSysLamK0 runAllSysLamKch ${MY_LIBS}
