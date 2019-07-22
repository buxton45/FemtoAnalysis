########################################################################
#
#              --- CAEN SpA - Front End Division ---
#
#   Created  :  November 2007      (Rel. 1.0)
#
#   Author: L. Colombini
#
########################################################################

EXE	=	V1190Readout

CC	=	gcc

COPTS	=	-fPIC -DLINUX -Wall
#COPTS	=	-g -fPIC -DLINUX -Wall

FLAGS	=	-Wall -s
#FLAGS	=	-Wall

DEPLIBS	=       -l CAENVME -lncurses -lm

LIBS	=

INCLUDEDIR =	-I.

OBJS	=	src/$(EXE).o src/keyb.o

INCLUDES =	src/keyb.h src/V1190.h

#########################################################################

all	:	$(EXE)

clean	:
		/bin/rm -f $(OBJS) $(EXE)

$(EXE)	:	$(OBJS)
		/bin/rm -f $(EXE)
		$(CC) $(FLAGS) -o $(EXE) $(OBJS) $(DEPLIBS)

$(OBJS)	:	$(INCLUDES) Makefile

%.o	:	%.c
		$(CC) $(COPTS) $(INCLUDEDIR) -c -o $@ $<

