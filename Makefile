#CC= $(firstword $(wildcard /usr/local/bin/gcc /usr/bin/gcc))
CC=gcc
CFLAGS= -std=c99 -pedantic -Wall -g -m64 -fstrict-aliasing
#CFLAGS= -std=c99 -pedantic -Wall -O3 -m64 -fstrict-aliasing -fopenmp -march=athlon64

LIBS= -L/usr/local/lib -ltiff -ljpeg -lpng -lz -lm
INCS= -I/usr/local/include

#-------------------------------------------------------------------------------

TOOLS = envremap envtoirr

all : $(TOOLS)

envremap : envremap.o
envtoirr : envtoirr.o

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
