#!/bin/bash
set -e

PROJECT_NAME=nervous
HEADERS=netarch.h,blasfuncs.h,nervous.h,module.map
echo `xcode-select -print-path`

#make test
make clean

buildIOS()
{
	ARCH=$1

	if [[ "${ARCH}" == "i386" || "${ARCH}" == "x86_64" ]]; then
		SDK=`xcrun --sdk iphonesimulator --show-sdk-path`
	else
		SDK=`xcrun --sdk iphoneos --show-sdk-path`
	fi
   
	echo "Building ${PROJECT_NAME} for ${ARCH} ${SDK}"

	make CXXFLAGS="-isysroot ${SDK} -arch ${ARCH}" LDFLAGS="-syslibroot ${SDK} -arch ${ARCH}" lib${PROJECT_NAME}.a 

	mv lib${PROJECT_NAME}.a tmp/lib${PROJECT_NAME}-${ARCH}.a

	make clean
}

echo "Cleaning up"
rm -rf lib

mkdir -p lib
mkdir -p lib/include
mkdir -p tmp

echo "Copying headers"
eval "cp src/{$HEADERS} lib/include"

./configure

buildIOS "arm64"
buildIOS "x86_64"
buildIOS "i386"
buildIOS  "armv7"
buildIOS  "armv7s"

echo "Building iOS libraries"
lipo tmp/*.a -create -output lib/lib${PROJECT_NAME}.a

echo "Cleaning up"
rm -rf tmp
echo "Done"