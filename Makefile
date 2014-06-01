#CC= $(firstword $(wildcard /usr/local/bin/gcc /usr/bin/gcc))
CC=gcc
CPP=g++-4.9
CFLAGS= -std=c99 -pedantic -Wall -g -m64 -fstrict-aliasing
#CFLAGS= -std=c99 -pedantic -Wall -O3 -m64 -fstrict-aliasing -fopenmp -march=athlon64
#CPPFLAGS= -mfpmath=sse -pedantic -Wall -O3 -m64 -fstrict-aliasing -fopenmp
CPPFLAGS= -g3 -fstrict-aliasing -pedantic

LIBS= -L/usr/local/lib -ltiff -ljpeg -lpng -lz -lm
INCS= -I/usr/local/include

#-------------------------------------------------------------------------------

TOOLS = envremap envtoirr envtospecular

all : $(TOOLS)

envremap : envremap.o
envtoirr : envtoirr.o

envtospecular.o: envtospecular.cpp
	$(CPP) $(CPPFLAGS) $(INCS) -o envtospecular.o -c envtospecular.cpp

envtospecular : envtospecular.o
	$(CPP) $(CPPFLAGS) $(LIBS) -o envtospecular envtospecular.o


#-------------------------------------------------------------------------------
# Define implicit rules building all tools.

%   : %.c

%.o : %.c
	$(CC) $(CFLAGS) $(INCS) -o $@ -c $<

%   : %.o
	$(CC) $(CFLAGS) $(LIBS) -o $@ $^

clean :
	rm -f $(TOOLS) $(UTILS) *.o

#-------------------------------------------------------------------------------
