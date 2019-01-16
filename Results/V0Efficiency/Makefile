ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
LEDNICKYFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/LednickyFitter
THERM_DIR = /home/jesse/Analysis/FemtoAnalysis/Therminator
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities
COULOMBFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter

LIBS   = $(shell gsl-config --libs) $(shell root-config --libs) -lMinuit
CFLAGS = $(shell gsl-config --cflags) $(shell root-config --cflags) -Wall -std=c++11 -lEG -I${ANALYZE_DIR} -I${LEDNICKYFITTER_DIR} -I${UTILITIES_DIR} -I${COULOMBFITTER_DIR} -I${THERM_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

RM = rm

MY_LIBS = $(addprefix build/, ParticleCoor.o Interpolator.o Types.o CanvasPartition.o PIDMapping.o ThermCommon.o ThermParticle.o ThermV0Particle.o ThermEvent.o ThermChargedResidual.o ThermPairAnalysis.o ThermSingleParticleAnalysis.o ThermFlowAnalysis.o ThermFlowCollection.o ThermEventsCollection.o SimpleThermAnalysis.o Therm3dCf.o ThermCf.o CfLite.o CfHeavy.o SuperpositionFitBgd.o SpherHarmonics.o CorrFctnDirectYlm.o)

BINARIES = runDrawEfficiencyPlots runBuildEfficiencyPlots

all: $(BINARIES)

runDrawEfficiencyPlots: ${MY_LIBS} DrawEfficiencyPlots.C
	g++ DrawEfficiencyPlots.C ${MY_LIBS} -o runDrawEfficiencyPlots $(LIBS) $(CFLAGS)

runBuildEfficiencyPlots: ${MY_LIBS} BuildEfficiencyPlots.C
	g++ BuildEfficiencyPlots.C ${MY_LIBS} -o runBuildEfficiencyPlots $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cxx ${LEDNICKYFITTER_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${COULOMBFITTER_DIR}/%.cxx ${COULOMBFITTER_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: ${THERM_DIR}/%.cxx ${THERM_DIR}/%.h
	g++ -c $< -I${INCDIR} $(LIBS) $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@


clean:
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
