VERSION=10.4
WSTPLINKDIR = /usr/local/Wolfram/Mathematica/10.4/SystemFiles/Links/WSTP/DeveloperKit
SYS = Linux-x86-64
CADDSDIR = ${WSTPLINKDIR}/${SYS}/CompilerAdditions

INCDIR = ${CADDSDIR}
LIBDIR = ${CADDSDIR}

EXTRALIBS = -lm -lpthread -lrt -lstdc++ -ldl -luuid
WSTPLIB = WSTP64i4

WSPREP = ${CADDSDIR}/wsprep

##------------------------------------

ANALYZE_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze
LEDNICKYFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/LednickyFitter
COULOMBFITTER_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter
UTILITIES_DIR = /home/jesse/Analysis/FemtoAnalysis/ProcessData/Utilities

LIBS	= -lcudart
CUDAINCLUDES = -I../../NVIDIA_CUDA-7.5_Samples/common/inc

LIBSC   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -pthread -Wno-deprecated-declarations -m64 -Wall -std=c++11 -I${ANALYZE_DIR} -I${LEDNICKYFITTER_DIR} -I${COULOMBFITTER_DIR} -I${UTILITIES_DIR} -g -O0 -fopenmp

MY_LIBS = $(addprefix build/, MathematicaSession.o WaveFunction.o CoulombFitter.o InterpolationHistograms.o FitParameter.o FitPartialAnalysis.o FitPairAnalysis.o FitSharedAnalyses.o CfLite.o CfHeavy.o Types.o Faddeeva.o ParallelTypes.o CoulombFitterParallel.o ParallelWaveFunction.o Interpolate.o InterpolateGPU.o )

all : runTest runInterp

runTest: ${MY_LIBS} test.C
	g++ test.C ${MY_LIBS} -I${INCDIR} -o runTest -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(LIBSC) $(CFLAGS)

runInterp: ${MY_LIBS} testInterp.C
	g++ testInterp.C ${MY_LIBS} -I${INCDIR} -o runInterp -L${LIBDIR} -l${WSTPLIB} ${EXTRALIBS} $(LIBS) $(LIBSC) $(CFLAGS)

build/%.o: ${ANALYZE_DIR}/%.cxx ${ANALYZE_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cxx ${LEDNICKYFITTER_DIR}/%.h
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${LEDNICKYFITTER_DIR}/%.cc ${LEDNICKYFITTER_DIR}/%.hh
	g++ -c $< $(CFLAGS) -o $@

build/%.o: ${COULOMBFITTER_DIR}/%.cxx ${COULOMBFITTER_DIR}/%.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: %.cxx %.h
	g++ -c $< -I${INCDIR} $(CFLAGS) -o $@

build/%.o: %.cu %.h
	nvcc -c -std=c++11 -arch=sm_50 -lineinfo -Xptxas -v --default-stream per-thread $< -o $@ -I${ANALYZE_DIR}

clean: 
	rm -f runTest runInterp ${MY_LIBS}

