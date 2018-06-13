ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
LEDNICKYFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/LednickyFitter
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities
COULOMBFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter

LIBS   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${LEDNICKYFITTER_DIR} -I${UTILITIES_DIR} -I${COULOMBFITTER_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

RM = rm

MY_LIBS = $(addprefix build/, ParticleCoor.o Interpolator.o Types.o CanvasPartition.o PIDMapping.o ThermCommon.o ThermParticle.o ThermV0Particle.o ThermEvent.o ThermChargedResidual.o ThermPairAnalysis.o ThermSingleParticleAnalysis.o ThermFlowAnalysis.o ThermEventsCollection.o SimpleThermAnalysis.o Therm3dCf.o ThermCf.o CfLite.o CfHeavy.o)

BINARIES = runTest runAnalysis runDrawTransforms runDrawPairFractions runDrawParentsMatrices runBuildLambdaValuesFromParentMatrices runProject3dRadii runProjectTransverseEmission runDrawPairSources runDrawCfs runDrawCfsv2 runCompareCfs runCompareCfsv2 runProject3dCfs runGetmT runRenameEventFiles runCompareBackgroundsAndFit runDrawIdentifiedv2

all: $(BINARIES)

runTest: ${MY_LIBS} test.C
	g++ test.C ${MY_LIBS} -o runTest $(LIBS) $(CFLAGS)

runAnalysis: ${MY_LIBS} RunAnalysis.C
	g++ RunAnalysis.C ${MY_LIBS} -o runAnalysis $(LIBS) $(CFLAGS)

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

runProjectTransverseEmission: ${MY_LIBS} ProjectTransverseEmission.C
	g++ ProjectTransverseEmission.C ${MY_LIBS} -o runProjectTransverseEmission $(LIBS) $(CFLAGS)

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

runGetmT: ${MY_LIBS} GetmT.C
	g++ GetmT.C ${MY_LIBS} -o runGetmT $(LIBS) $(CFLAGS)

runRenameEventFiles: ${MY_LIBS} RenameEventFiles.C
	g++ RenameEventFiles.C ${MY_LIBS} -o runRenameEventFiles $(LIBS) $(CFLAGS)

runCompareBackgroundsAndFit: ${MY_LIBS} CompareBackgroundsAndFit.C
	g++ CompareBackgroundsAndFit.C ${MY_LIBS} -o runCompareBackgroundsAndFit $(LIBS) $(CFLAGS)

runDrawIdentifiedv2: ${MY_LIBS} DrawIdentifiedv2.C
	g++ DrawIdentifiedv2.C ${MY_LIBS} -o runDrawIdentifiedv2 $(LIBS) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cxx ${LEDNICKYFITTER_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${COULOMBFITTER_DIR}/%.cxx ${COULOMBFITTER_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@


clean:
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
