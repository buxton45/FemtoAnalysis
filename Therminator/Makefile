ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities
COULOMBFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter

LIBS   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${UTILITIES_DIR} -I${COULOMBFITTER_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

RM = rm

MY_LIBS = $(addprefix build/, ParticleCoor.o Interpolator.o Types.o PIDMapping.o ThermCommon.o ThermParticle.o ThermV0Particle.o ThermEvent.o ThermChargedResidual.o ThermPairAnalysis.o ThermSingleParticleAnalysis.o ThermEventsCollection.o SimpleThermAnalysis.o Therm3dCf.o)

BINARIES = runTest runTest2 runDrawTransforms runDrawPairFractions runDrawParentsMatrices runBuildLambdaValuesFromParentMatrices runProject3dRadii runDrawPairSources runDrawCfs runDrawCfsv2 runCompareCfs runCompareCfsv2 runProject3dCfs

all: $(BINARIES)

runTest: ${MY_LIBS} test.C
	g++ test.C ${MY_LIBS} -o runTest $(LIBS) $(CFLAGS)

runTest2: ${MY_LIBS} test2.C
	g++ test2.C ${MY_LIBS} -o runTest2 $(LIBS) $(CFLAGS)

runDrawTransforms: ${MY_LIBS} DrawTransformMatrices.C
	g++ DrawTransformMatrices.C ${MY_LIBS} -o runDrawTransforms $(LIBS) $(CFLAGS)

runDrawPairFractions: ${MY_LIBS} DrawPairFractions.C
	g++ DrawPairFractions.C ${MY_LIBS} -o runDrawPairFractions $(LIBS) $(CFLAGS)

runDrawParentsMatrices: ${MY_LIBS} DrawParentsMatrices.C
	g++ DrawParentsMatrices.C ${MY_LIBS} -o runDrawParentsMatrices $(LIBS) $(CFLAGS)


runBuildLambdaValuesFromParentMatrices: ${MY_LIBS} BuildLambdaValuesFromParentMatrices.C
	g++ BuildLambdaValuesFromParentMatrices.C ${MY_LIBS} -o runBuildLambdaValuesFromParentMatrices $(LIBS) $(CFLAGS)

runProject3dRadii: ${MY_LIBS} Project3dRadiiMatrices.C
	g++ Project3dRadiiMatrices.C ${MY_LIBS} -o runProject3dRadii $(LIBS) $(CFLAGS)

runDrawPairSources: ${MY_LIBS} DrawPairSources.C
	g++ DrawPairSources.C ${MY_LIBS} -o runDrawPairSources $(LIBS) $(CFLAGS)

runDrawCfs: ${MY_LIBS} DrawCorrelationFunctions.C
	g++ DrawCorrelationFunctions.C ${MY_LIBS} -o runDrawCfs $(LIBS) $(CFLAGS)

runDrawCfsv2: ${MY_LIBS} DrawCorrelationFunctionsv2.C
	g++ DrawCorrelationFunctionsv2.C ${MY_LIBS} -o runDrawCfsv2 $(LIBS) $(CFLAGS)

runCompareCfs: ${MY_LIBS} CompareCorrelationFunctions.C
	g++ CompareCorrelationFunctions.C ${MY_LIBS} -o runCompareCfs $(LIBS) $(CFLAGS)

runCompareCfsv2: ${MY_LIBS} CompareCorrelationFunctionsv2.C
	g++ CompareCorrelationFunctionsv2.C ${MY_LIBS} -o runCompareCfsv2 $(LIBS) $(CFLAGS)

runProject3dCfs: ${MY_LIBS} Project3dCfs.C
	g++ Project3dCfs.C ${MY_LIBS} -o runProject3dCfs $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${COULOMBFITTER_DIR}/%.cxx ${COULOMBFITTER_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@


clean:
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
