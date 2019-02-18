##gsl-config stuff needed to use Adam's SpherHarmonics and CorrFctnDirectYlm
LIBS   = $(shell gsl-config --libs) $(shell root-config --libs) -lMinuit
CFLAGS = $(shell gsl-config --cflags) $(shell root-config --cflags) -Wall -std=c++11 -I${ANALYZE_DIR} -I${LEDNICKYFITTER_DIR} -I${UTILITIES_DIR} -I${COULOMBFITTER_DIR} -g -O0
#IFLAGS = -I$(ROOTSYS)/include
#.PHONY: clean

RM = rm

MY_LIBS = $(addprefix build/, Faddeeva.o)

BINARIES = Monte_Carlo_test

all: $(BINARIES)

Monte_Carlo_test: ${MY_LIBS} Monte_Carlo_test.cpp
	g++ -Wall -o Monte_Carlo_test Monte_Carlo_test.cpp ${MY_LIBS} $(LIBS) $(CFLAGS) -lgsl -lgslcblas

build/%.o: %.cc %.hh
	g++ -c $< $(CFLAGS) -o $@

clean:
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
