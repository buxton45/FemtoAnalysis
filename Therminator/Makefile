ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${UTILITIES_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

MY_LIBS = $(addprefix build/, Types.o PIDMapping.o ThermParticle.o ThermV0Particle.o ThermEvent.o ThermEventsCollection.o)


all: runTest runDrawTransforms runDrawPairFractions runDrawParentsMatrices runBuildLambdaValuesFromParentMatrices

runTest: ${MY_LIBS} test.C
	g++ test.C ${MY_LIBS} -o runTest $(LIBS) $(CFLAGS)

runDrawTransforms: ${MY_LIBS} DrawTransformMatrices.C
	g++ DrawTransformMatrices.C ${MY_LIBS} -o runDrawTransforms $(LIBS) $(CFLAGS)

runDrawPairFractions: ${MY_LIBS} DrawPairFractions.C
	g++ DrawPairFractions.C ${MY_LIBS} -o runDrawPairFractions $(LIBS) $(CFLAGS)

runDrawParentsMatrices: ${MY_LIBS} DrawParentsMatrices.C
	g++ DrawParentsMatrices.C ${MY_LIBS} -o runDrawParentsMatrices $(LIBS) $(CFLAGS)


runBuildLambdaValuesFromParentMatrices: ${MY_LIBS} BuildLambdaValuesFromParentMatrices.C
	g++ BuildLambdaValuesFromParentMatrices.C ${MY_LIBS} -o runBuildLambdaValuesFromParentMatrices $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@


clean:
	rm -f runTest ${MY_LIBS}
