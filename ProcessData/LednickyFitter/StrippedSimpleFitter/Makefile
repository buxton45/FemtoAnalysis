RM = rm
######################################################
LIBS   = $(shell root-config --libs) -lMinuit
CFLAGS = $(shell root-config --cflags) -Wall -std=c++11 -g -O0


MY_LIBS = StrippedSimpleFitter.o Faddeeva.o

BINARIES = runFit

all: $(BINARIES)

runFit: ${MY_LIBS} RunStrippedSimpleFitter.C
	g++ RunStrippedSimpleFitter.C ${MY_LIBS} -o runFit $(LIBS) $(CFLAGS)


%.o: %.cxx %.h
	g++ -c $< $(CFLAGS) -o $@

%.o: %.cc %.hh
	g++ -c $< $(CFLAGS) -o $@

clean:
	@ ${RM} -rf *.o $(BINARIES) ${MY_LIBS}
