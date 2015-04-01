AM_CXXFLAGS = -g -pedantic -Wall -std=c++11 -Iinclude -Isrc
AM_LDFLAGS =

all: libnervous.a

lib_LIBRARIES = libnervous.a

libnervous_a_SOURCES = src/blasfuncs.cpp src/netarch.cpp src/nettools.cpp src/mathutil.cpp src/layer.cpp src/net.cpp src/training.cpp

ACLOCAL_AMFLAGS = -I m4 --install
EXTRA_DIST = m4/NOTES