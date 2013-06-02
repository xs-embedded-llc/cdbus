CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0)
PROJECT( cdbus  C )

# Add an additional modules path
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules")
set(FIND_LIBRARY_USE_LIB64_PATHS TRUE)


# Define the CDBUS root directory
set(CDBUS_ROOT_DIR ${PROJECT_SOURCE_DIR})
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CDBUS_ROOT_DIR}/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE}/lib )
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CDBUS_ROOT_DIR}/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE}/lib )
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CDBUS_ROOT_DIR}/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE}/bin )

set(CDBUS_MAJOR_VERSION 1)
set(CDBUS_MINOR_VERSION 0)
set(CDBUS_RELEASE_VERSION 0)
set(CDBUS_VERSION_STRING ${CDBUS_MAJOR_VERSION}.${CDBUS_MINOR_VERSION}.${CDBUS_RELEASE_VERSION})

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
  cmake_policy(SET CMP0007 OLD)
endif(COMMAND cmake_policy)

set(CDBUS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB CDBUS_SRC_FILES "${CDBUS_SRC_DIR}/*.c")

set(CDBUS_MAIN_LOOP_EV_SRC_FILES "${CDBUS_SRC_DIR}/main-loop/main-loop-ev.c")
set(CDBUS_MAIN_LOOP_GLIB_SRC_FILES "${CDBUS_SRC_DIR}/main-loop/main-loop-glib.c")

# The "core" CDBUS library
add_library(CDBUS_STATIC STATIC ${CDBUS_SRC_FILES})
set_target_properties(CDBUS_STATIC PROPERTIES OUTPUT_NAME cdbus)
add_library(CDBUS_SHARED SHARED ${CDBUS_SRC_FILES})
set_target_properties(CDBUS_SHARED PROPERTIES OUTPUT_NAME cdbus)

set_target_properties(CDBUS_SHARED PROPERTIES 
                      SOVERSION "${CDBUS_MAJOR_VERSION}"
                      VERSION "${CDBUS_VERSION_STRING}")
# This forces the builds to serialize
add_dependencies(CDBUS_SHARED CDBUS_STATIC)

                      
# The libev main loop library
if( NOT CDBUS_NO_GLIB_LOOP )
    add_library(CDBUS_MAIN_LOOP_EV_STATIC STATIC ${CDBUS_MAIN_LOOP_EV_SRC_FILES})
    set_target_properties(CDBUS_MAIN_LOOP_EV_STATIC PROPERTIES OUTPUT_NAME cdbus-ev)
    add_library(CDBUS_MAIN_LOOP_EV_SHARED SHARED ${CDBUS_MAIN_LOOP_EV_SRC_FILES})
    set_target_properties(CDBUS_MAIN_LOOP_EV_SHARED PROPERTIES OUTPUT_NAME cdbus-ev)
    
    set_target_properties(CDBUS_MAIN_LOOP_EV_SHARED PROPERTIES 
                          SOVERSION "${CDBUS_MAJOR_VERSION}"
                          VERSION "${CDBUS_VERSION_STRING}")
    # This forces the builds to serialize
    add_dependencies(CDBUS_MAIN_LOOP_EV_SHARED CDBUS_MAIN_LOOP_EV_STATIC)
endif( NOT CDBUS_NO_GLIB_LOOP )

                      
# The glib main loop library
if( NOT CDBUS_NO_LIBEV_LOOP )
    add_library(CDBUS_MAIN_LOOP_GLIB_STATIC STATIC ${CDBUS_MAIN_LOOP_GLIB_SRC_FILES})
    set_target_properties(CDBUS_MAIN_LOOP_GLIB_STATIC PROPERTIES OUTPUT_NAME cdbus-glib)
    add_library(CDBUS_MAIN_LOOP_GLIB_SHARED SHARED ${CDBUS_MAIN_LOOP_GLIB_SRC_FILES})
    set_target_properties(CDBUS_MAIN_LOOP_GLIB_SHARED PROPERTIES OUTPUT_NAME cdbus-glib)
    
    set_target_properties(CDBUS_MAIN_LOOP_GLIB_SHARED PROPERTIES 
                          SOVERSION "${CDBUS_MAJOR_VERSION}"
                          VERSION "${CDBUS_VERSION_STRING}")
    # This forces the builds to serialize
    add_dependencies(CDBUS_MAIN_LOOP_GLIB_SHARED CDBUS_MAIN_LOOP_GLIB_STATIC)
endif( NOT CDBUS_NO_LIBEV_LOOP )



# Configure the compiler settings
include(CompilerSettings)

set(INCLUDE_DIRS "${CDBUS_ROOT_DIR}/include")
include_directories( "${INCLUDE_DIRS}" )

# Pull in the pkg-config scripts
include(FindPkgConfig)

pkg_check_modules(DBUSLIB_PKG REQUIRED "dbus-1>=1.4.0")
include_directories(${DBUSLIB_PKG_INCLUDE_DIRS})
link_directories(${DBUSLIB_PKG_LIBRARY_DIRS})

if( NOT CDBUS_NO_LIBEV_LOOP )
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
    
    target_link_libraries(CDBUS_MAIN_LOOP_EV_SHARED ${LIBEV_PKG_LIBRARIES} "cdbus")
    target_link_libraries(CDBUS_MAIN_LOOP_EV_STATIC ${LIBEV_PKG_LIBRARIES} "cdbus")
endif( NOT CDBUS_NO_LIBEV_LOOP )

if( NOT CDBUS_NO_GLIB_LOOP )
    pkg_check_modules(GLIB_PKG REQUIRED "glib-2.0>=2.0.0")
    include_directories(${GLIB_PKG_INCLUDE_DIRS})
    link_directories(${GLIB_PKG_LIBRARY_DIRS})
    
    target_link_libraries(CDBUS_MAIN_LOOP_GLIB_SHARED ${GLIB_PKG_LIBRARIES} "cdbus")
    target_link_libraries(CDBUS_MAIN_LOOP_GLIB_STATIC ${GLIB_PKG_LIBRARIES} "cdbus")
endif( NOT CDBUS_NO_GLIB_LOOP )

target_link_libraries(CDBUS_SHARED ${DBUSLIB_PKG_LIBRARIES})
target_link_libraries(CDBUS_STATIC ${DBUSLIB_PKG_LIBRARIES})

LINK_DIRECTORIES(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Installation setup
set(INSTALL_TARGETS_DEFAULT_ARGS
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY "${INCLUDE_DIRS}/cdbus" DESTINATION include
        PATTERN "main-loop-*.h" EXCLUDE)
install(TARGETS CDBUS_SHARED LIBRARY DESTINATION lib)
install(TARGETS CDBUS_STATIC ARCHIVE DESTINATION lib)

if( NOT CDBUS_NO_GLIB_LOOP )
    install(FILES "${INCLUDE_DIRS}/cdbus/main-loop-glib.h" DESTINATION include/cdbus)
    install(TARGETS CDBUS_MAIN_LOOP_GLIB_SHARED LIBRARY DESTINATION lib)
    install(TARGETS CDBUS_MAIN_LOOP_GLIB_STATIC ARCHIVE DESTINATION lib)
endif(NOT CDBUS_NO_GLIB_LOOP)

if( NOT CDBUS_NO_LIBEV_LOOP )
    install(FILES "${INCLUDE_DIRS}/cdbus/main-loop-ev.h" DESTINATION include/cdbus)
    install(TARGETS CDBUS_MAIN_LOOP_EV_SHARED LIBRARY DESTINATION lib)
    install(TARGETS CDBUS_MAIN_LOOP_EV_STATIC ARCHIVE DESTINATION lib)
endif( NOT CDBUS_NO_LIBEV_LOOP )

# Uninstall target
configure_file(
    "${PROJECT_SOURCE_DIR}/CMakeModules/CMakeUninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY
)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE cdbus)

# Create a cdbus-config.cmake file for the use from the build tree
set(CDBUS_INCLUDE_DIRS "${INCLUDE_DIRS}")
set(CDBUS_LIB_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
# Not used right now (export depends?)
#set(CDBUS_CMAKE_DIR "${PROJECT_BINARY_DIR}")
configure_file(cdbus-config.cmake.in
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/cdbus/cdbus-config.cmake" @ONLY)
configure_file(cdbus-config-version.cmake.in
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/cdbus/cdbus-config-version.cmake" @ONLY)

 
# Create a cdbus-config.cmake file for the use from the install tree
# and install it
set(CDBUS_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")
set(CDBUS_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
#set(OCTOMAP_CMAKE_DIR "${INSTALL_DATA_DIR}/FooBar/CMake")
configure_file(cdbus-config.cmake.in
  "${PROJECT_BINARY_DIR}/InstallFiles/cdbus-config.cmake" @ONLY)
configure_file(cdbus-config-version.cmake.in
  "${PROJECT_BINARY_DIR}/InstallFiles/cdbus-config-version.cmake" @ONLY)
install(FILES
  "${PROJECT_BINARY_DIR}/InstallFiles/cdbus-config.cmake"
  "${PROJECT_BINARY_DIR}/InstallFiles/cdbus-config-version.cmake"
  DESTINATION lib/cmake/cdbus/)

# Write pkgconfig-file:
include(InstallPkgConfigFile)
install_pkg_config_file(cdbus
    DESCRIPTION "Simple 'C' binding to D-Bus library with main loop integration."
    CFLAGS
    LIBS -lcdbus
    REQUIRES dbus-1
    VERSION ${CDBUS_MAJOR_VERSION}.${CDBUS_MINOR_VERSION}.${CDBUS_RELEASE_VERSION})

# Documentation
FIND_PACKAGE(Doxygen)
IF(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/cdbus.dox.in ${CMAKE_CURRENT_BINARY_DIR}/docs/cdbus.dox @ONLY)
  ADD_CUSTOM_TARGET(pub-docs ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/docs/cdbus.dox
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating documentation (Doxygen)...")
ENDIF(DOXYGEN_FOUND)

# Make package release from source (must be done on the GIT source tree)
SET(CDBUS_PKG_NAME "${PROJECT_NAME}-${CDBUS_VERSION_STRING}.tar.gz")
SET(DIST_DIR "${CMAKE_BINARY_DIR}/dist-${PROJECT_NAME}")
ADD_CUSTOM_TARGET("dist-${PROJECT_NAME}"
    rm -rf "${DIST_DIR}" "${CMAKE_BINARY_DIR}/${CDBUS_PKG_NAME}"
    COMMAND mkdir "${DIST_DIR}"
    COMMAND git clone --depth 1 ssh://git@git.xs-embedded.us:2222/cdbus.git "${DIST_DIR}/${PROJECT_NAME}"
    COMMAND tar -czf "${CMAKE_BINARY_DIR}/${CDBUS_PKG_NAME}"
            -C "${DIST_DIR}" --exclude-vcs --exclude="arm*" --exclude="x86*"
            --exclude=".cproject" --exclude=".project" --exclude=".settings" "${PROJECT_NAME}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    # Cleanup so that there is no copy in the source dir
    COMMAND rm -rf "${DIST_DIR}"
)


# Add a custom target to install/uninstall public documentation
ADD_CUSTOM_TARGET(install-docs
    install -d "${CMAKE_INSTALL_PREFIX}/share/doc/${PROJECT_NAME}"
    COMMAND cp -rf "${CMAKE_CURRENT_BINARY_DIR}/docs/api/*" "${CMAKE_INSTALL_PREFIX}/share/doc/${PROJECT_NAME}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
ADD_CUSTOM_TARGET(uninstall-docs
    rm -rf "${CMAKE_INSTALL_PREFIX}/share/doc/${PROJECT_NAME}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
ADD_DEPENDENCIES(install-docs pub-docs)
ADD_DEPENDENCIES(uninstall uninstall-docs)

  
# Needs to be last statement:
INCLUDE(CPackSettings)

# Finished:
MESSAGE ("\n")
MESSAGE (STATUS "Compile ${PROJECT_NAME} using: make")
MESSAGE (STATUS "Install ${PROJECT_NAME} using: make install")
MESSAGE (STATUS "Uninstall ${PROJECT_NAME} using: make uninstall")
MESSAGE (STATUS "Compile public API documentation using: make pub-docs")
MESSAGE (STATUS "Install public API  documentation using: make install-docs")
MESSAGE (STATUS "Uninstall public API  documentation using: make uninstall-docs")
MESSAGE (STATUS "Create ${PROJECT_NAME} source distribution using: make dist-${PROJECT_NAME}")
MESSAGE (STATUS " (be sure to set the correct CMAKE_INSTALL_PREFIX before)\n")


