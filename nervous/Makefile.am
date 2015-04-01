AM_CXXFLAGS = -g -pedantic -Wall -std=c++11 -Iinclude -Isrc
AM_LDFLAGS =

all: libnervous.a

lib_LIBRARIES = libnervous.a
libnervous_a_SOURCES = src/blasfuncs.cpp src/netarch.cpp src/nettools.cpp src/math.cpp src/foo.cpp

ACLOCAL_AMFLAGS = -I m4 --install
EXTRA_DIST = m4/NOTES