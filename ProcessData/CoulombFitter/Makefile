# This makefile can be used to build all or some of the sample
# programs.  To build all of them, use the command
# 'make all'.  To build one, say addtwo, use the command
# 'make addtwo'.

# Portions of this makefile require the use of GNU make.
# see http://www.gnu.org/software/make for more information.

VERSION=10.4
WSTPLINKDIR = /usr/local/Wolfram/Mathematica/10.4/SystemFiles/Links/WSTP/DeveloperKit
SYS = Linux-x86-64
CADDSDIR = ${WSTPLINKDIR}/${SYS}/CompilerAdditions

INCDIR = ${CADDSDIR}
LIBDIR = ${CADDSDIR}

EXTRALIBS = -lm -lpthread -lrt -lstdc++ -ldl -luuid
WSTPLIB = WSTP64i4

WSPREP = ${CADDSDIR}/wsprep
RM = rm

ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
LEDNICKYFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/LednickyFitter
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS   = $(shell root-config --libs) -lMinuit
#CFLAGS = $(shell root-config --cflags) -Wall -g -O0
CFLAGS = $(shell root-config --cflags) -pthread -Wno-deprecated-declarations -m64 -Wall -I${ANALYZE_DIR} -I${LEDNICKYFITTER_DIR} -I${UTILITIES_DIR} -g -O0 -fopenmp
#CFLAGS = -pthread -Wno-deprecated-declarations -m64 -Wall -I${ANALYZE_DIR} -I${LEDNICKYFITTER_DIR} -g -O0 -I/opt/alice/root/alice_v5-34-30/inst/include #because I have to use c++ 99 

MY_LIBS = $(addprefix build/, MathematicaSession.o WaveFunction.o CoulombFitter.o SimpleFitter.o InterpolationHistograms.o FitChi2Histograms.o FitParameter.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o CfLite.o CfHeavy.o Types.o Faddeeva.o)

BINARIES = runTest runFullTest runBuildInterpolationHistograms runFitLambda runSimpleFitter runProtonCascade runManualFitting

all : $(BINARIES)

runTest: ${MY_LIBS} test.C	
	${CXX} test.C ${MY_LIBS} -I${INCDIR} -o runTest -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS) -o $@

runFullTest: ${MY_LIBS} BuildFullTest.C	
	${CXX} BuildFullTest.C ${MY_LIBS} -I${INCDIR} -o runFullTest -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS) -o $@

runBuildInterpolationHistograms: ${MY_LIBS} BuildInterpolationHistograms.C	
	${CXX} BuildInterpolationHistograms.C ${MY_LIBS} -I${INCDIR} -o runBuildInterpolationHistograms -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS) -o $@

runFitLambda: ${MY_LIBS} BuildFitLambda.C	
	${CXX} BuildFitLambda.C ${MY_LIBS} -I${INCDIR} -o runFitLambda -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS) -o $@

runSimpleFitter: ${MY_LIBS} SimpleTest.C	
	${CXX} SimpleTest.C ${MY_LIBS} -I${INCDIR} -o runSimpleFitter -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS) -o $@

runProtonCascade: ${MY_LIBS} BuildProtonCascade.C	
	${CXX} BuildProtonCascade.C ${MY_LIBS} -I${INCDIR} -o runProtonCascade -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS) -o $@

runManualFitting: ${MY_LIBS} ManualFitting.C	
	${CXX} ManualFitting.C ${MY_LIBS} -I${INCDIR} -o runManualFitting -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(CFLAGS) -o $@

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cxx ${LEDNICKYFITTER_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cc ${LEDNICKYFITTER_DIR}/%.hh
	g++ -c $< $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

clean :
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
