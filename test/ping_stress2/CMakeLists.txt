CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0)
PROJECT( ping_stress  C )

# Add an additional modules path
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules")
set(FIND_LIBRARY_USE_LIB64_PATHS TRUE)

include(FindPkgConfig)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
  cmake_policy(SET CMP0007 OLD)
endif(COMMAND cmake_policy)

set(PING_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB PING_SRC_FILES "${PING_SRC_DIR}/*.c")

add_executable(ping_stress2 ${PING_SRC_FILES})

# Configure the compiler settings
include(CompilerSettings)

pkg_check_modules(DBUSLIB_PKG REQUIRED "dbus-1>=1.4.0")
include_directories(${DBUSLIB_PKG_INCLUDE_DIRS})
link_directories(${DBUSLIB_PKG_LIBRARY_DIRS})


pkg_check_modules(LIBEV_PKG "ev")
if( LIBEV_PKG_FOUND )
    include_directories(${LIBEV_PKG_INCLUDE_DIRS})
    link_directories(${LIBEV_PKG_LIBRARY_DIRS})
else( LIBEV_PKG_FOUND )
    # Search for the library in the usual places
    find_library( LIBEV_LIB ev )
    if( "${LIBEV_LIB}" STREQUAL "LIBEV_LIB-NOTFOUND" )
        message(FATAL_ERROR "Cannot find libev in the library search path")
    else( "${LIBEV_LIB}" STREQUAL "LIBEV_LIB-NOTFOUND" )
        find_path(LIBEV_PKG_INCLUDE_DIRS ev.h)
        include_directories(${LIBEV_PKG_INCLUDE_DIRS})
        get_filename_component(LIBEV_PKG_LIBRARY_DIRS ${LIBEV_LIB} PATH)
        link_directories(${LIBEV_PKG_LIBRARY_DIRS})
        set(LIBEV_PKG_LIBRARIES "ev")
        message(STATUS "Will link with libev found here: ${LIBEV_LIB}")
        message(STATUS "LIBEV_PKG_INCLUDE_DIRS: ${LIBEV_PKG_INCLUDE_DIRS}")
        message(STATUS "LIBEV_PKG_LIBRARY_DIRS: ${LIBEV_PKG_LIBRARY_DIRS}")
    endif( "${LIBEV_LIB}" STREQUAL "LIBEV_LIB-NOTFOUND" )
endif( LIBEV_PKG_FOUND )


pkg_check_modules(CDBUS_PKG "cdbus")
if( CDBUS_PKG_FOUND )
    include_directories(${CDBUS_PKG_INCLUDE_DIRS})
    link_directories(${CDBUS_PKG_LIBRARY_DIRS})
else( CDBUS_PKG_FOUND )
    # Search for the library in the usual places
    find_library( CDBUS_LIB "cdbus" )
    if( "${CDBUS_LIB}" STREQUAL "CDBUS_LIB-NOTFOUND" )
        message(FATAL_ERROR "Cannot find libcdbus in the library search path")
    else( "${CDBUS_LIB}" STREQUAL "CDBUS_LIB-NOTFOUND" )
        message(STATUS "Will link with libcdbus found here: "${CDBUS_LIB})
    endif( "${CDBUS_LIB}" STREQUAL "CDBUS_LIB-NOTFOUND" )
endif( CDBUS_PKG_FOUND )

# See if a specific Lua version has already been specified
if( NOT DEFINED PING_LUA_VERSION )
    # The default is to use 5.1
    set(PING_LUA_VERSION "5.1")
endif( NOT DEFINED PING_LUA_VERSION )

MESSAGE(STATUS "Searching for Lua ${PING_LUA_VERSION}")
find_package(Lua ${PING_LUA_VERSION} REQUIRED)
if ( "${LUA_VERSION_STRING}" STREQUAL "LUA_VERSION_STRING-NOTFOUND" )
    set(LUA_VERSION_STRING "${PING_LUA_VERSION}")
endif ( "${LUA_VERSION_STRING}" STREQUAL "LUA_VERSION_STRING-NOTFOUND" )

MESSAGE(STATUS "Found Lua version ${LUA_VERSION_STRING}")


include_directories(${LUA_INCLUDE_DIR})
target_link_libraries(ping_stress2 ${DBUSLIB_PKG_LIBRARIES}
                                   ${LIBEV_PKG_LIBRARIES}
                                   ${CDBUS_PKG_LIBRARIES}
                                   ${LUA_LIBRARIES})

LINK_DIRECTORIES(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

# Finished:
MESSAGE ("\n")
MESSAGE (STATUS "Compile ${PROJECT_NAME} using: make")



