CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0)

FILE(GLOB src_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

add_library(cdbus_static STATIC ${src_files})
set_target_properties(cdbus_static PROPERTIES OUTPUT_NAME cdbus)
set_target_properties(cdbus_static PROPERTIES PREFIX "")

add_library(cdbus_shared SHARED ${src_files})
set_target_properties(cdbus_shared PROPERTIES OUTPUT_NAME cdbus)
set_target_properties(cdbus_shared PROPERTIES PREFIX "")

# This forces the builds to serialize
add_dependencies(cdbus_shared cdbus_static)

add_definitions( -D_GNU_SOURCE )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

include(FindPkgConfig)
pkg_check_modules(DBUSLIB_PKG REQUIRED "dbus-1")
include_directories(${DBUSLIB_PKG_INCLUDE_DIRS})
link_directories(${DBUSLIB_PKG_LIBRARY_DIRS})
target_link_libraries(cdbus_shared ${DBUSLIB_PKG_LIBRARIES})
target_link_libraries(cdbus_static ${DBUSLIB_PKG_LIBRARIES})

get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("Include dirs = ${inc_dirs}")

IF(NOT DEFINED INSTALL_LIBS)
	SET(INSTALL_LIBS "/usr/lib")
ENDIF(NOT DEFINED INSTALL_LIBS)

IF(NOT DEFINED INSTALL_INCL)
	SET(INSTALL_INCL "/usr/include/cdbus")
ENDIF(NOT DEFINED INSTALL_INCL)

# Install libraries
install(TARGETS cdbus_shared DESTINATION ${INSTALL_LIBS})
install(TARGETS cdbus_static DESTINATION ${INSTALL_LIBS})

# Install headers
FILE(GLOB hdr_files "${CMAKE_CURRENT_SOURCE_DIR}/include/cdbus/*.h")
INSTALL(FILES ${hdr_files} DESTINATION ${INSTALL_INCL})


