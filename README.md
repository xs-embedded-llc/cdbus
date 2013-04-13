# "C" D-Bus (CDBUS) Library
---
## Introduction

The *C* D-Bus library (CDBUS) is a simple binding (in *C*) to the D-Bus reference library. It does **not** attempt to wrap and abstract the entire reference library but is intended, instead, to compliment the functions and features provided by it. To that end, CDBUS does provide several useful abstraction (Timer, Watch, Dispatcher, Interface, etc...) that simplify implementing D-Bus clients and services from *C* without incurring the need for a heavier-weight framework (e.g. GLib/GObject, QT, etc...). It also provides a *main loop* courtesy of an underlying [libev](http://software.schmorp.de/pkg/libev.html) based main loop. 

## Dependencies

CDBUS depends on the following external libraries:

* [D-Bus reference Library](http://dbus.freedesktop.org/releases/dbus/) (version > 1.4.X)
* [libev](http://software.schmorp.de/pkg/libev.html) (version >= 4.00)
* [Doxygen](http://www.doxygen.org/) Necessary for building documentation.
* [CMake](http://www.cmake.org/) (version >= 2.6.0) Necessary for building.


## Building

Before building CDBUS all of it's [dependencies](#Dependencies) must be built and installed in their proper locations. The CDBUS build is based on CMake and a convenient script (*build_host.sh*) is provided to simplify the build process. To build the library (once the dependencies are built and installed) execute the following command:

	# ./build_host.sh

By default a *release* version of the library is built. A "debug" option can be specified as the first argument to make a debug version of the library:

	# ./build_host.sh debug

There is also a script to build a cross-compiled version of the library. This needs to up modified for the particular build environment.

The library installation path prefix can be modified by passing in a CMake variable on the command line:

	# ./build_host.sh -DCMAKE_INSTALL_PREFIX=/usr/local

The default installation prefix is typically */usr/local*. 

The host build script will create a directory in the project *root* directory with the machine name with a suffix indicating whether this is a *Release* or *Debug* build. So on an x86 64-Bit Linux machine the resulting directory might be:

**x86_64-Release**

Inside this directory will be all the by-products of the build. If you move into the resulting build directory several additional *make* targets are available.

To install the library type:

	# make install

To uninstall the library type:

	# make uninstall

To generate documention type:

	# make docs

To generate a source distribution type:

	# make dist-cdbus

And of course to *clean* the build you can simple erase the build directory from the project root:

	# rm -rf x86_64-Release

## License ##

The CDBUS library itself is licensed under the Apache License Version 2.0. See <a href="./LICENSE">LICENSE</a> in the source distribution for additional details. Additional terms may apply since CDBUS does have other dependencies. The intent, however, was to utilize components that only have compatible licenses and support both open source and propriety (closed) source applications.


