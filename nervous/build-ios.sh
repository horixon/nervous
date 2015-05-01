#!/bin/bash
#
#  build-ios.sh
#  Neuralnet
#
#  data-tools ver. 01
#
#  Copyright (c) Microsoft Corporation
#
#  All rights reserved.
#
#  MIT License
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ""Software""), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

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