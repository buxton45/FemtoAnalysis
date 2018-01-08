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

##------------------------------------

ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
LEDNICKYFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/LednickyFitter
COULOMBFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS	= -lcudart
CUDAINCDIR = /usr/local/cuda/include
CUDALIBDIR = /usr/local/cuda/lib64

LIBSC   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -pthread -Wno-deprecated-declarations -m64 -Wall -std=c++11 -I${ANALYZE_DIR} -I${LEDNICKYFITTER_DIR} -I${COULOMBFITTER_DIR} -I${UTILITIES_DIR} -g -O0 -fopenmp

MY_LIBS = $(addprefix build/, MathematicaSession.o WaveFunction.o Interpolator.o ChargedResidualCf.o SimpleChargedResidualCf.o NeutralResidualCf.o ResidualCollection.o CoulombFitter.o LednickyFitter.o InterpolationHistograms.o FitChi2Histograms.o FitParameter.o BackgroundFitter.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o CfLite.o CfHeavy.o Types.o Types_LambdaValues.o Types_FitParamValues.o Faddeeva.o ParallelTypes.o CoulombFitterParallel.o ParallelWaveFunction.o Interpolate.o InterpolateGPU.o )

BINARIES = runTest runInterp

all : $(BINARIES)

runTest: ${MY_LIBS} test.C
	g++ test.C ${MY_LIBS} -I${INCDIR} -o runTest -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(LIBSC) $(CFLAGS) -I${CUDAINCDIR} -L${CUDALIBDIR}

runInterp: ${MY_LIBS} testInterp.C
	g++ testInterp.C ${MY_LIBS} -I${INCDIR} -o runInterp -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(LIBSC) $(CFLAGS) -I${CUDAINCDIR} -L${CUDALIBDIR}

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cxx ${LEDNICKYFITTER_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cc ${LEDNICKYFITTER_DIR}/%.hh
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${COULOMBFITTER_DIR}/%.cxx ${COULOMBFITTER_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -I${CUDAINCDIR} -L${CUDALIBDIR} -o $@

build/%.o: %.cu %.h
	nvcc -c -std=c++11 -arch=sm_50 -lineinfo -Xptxas -v --default-stream per-thread $< -o $@ -I${ANALYZE_DIR} -I${CUDAINCDIR} -L${CUDALIBDIR}

clean: 
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}

