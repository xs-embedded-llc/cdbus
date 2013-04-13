/*******************************************************************************
 * 
 * Project         cdbus
 * (c) Copyright   2012 XS-Embedded LLC
 *                 All rights reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *******************************************************************************
 *******************************************************************************
 * @file           macros.h        
 * @author         Glenn Schmottlach
 * @brief          Definition of common macros
 *******************************************************************************
 */

#ifndef CDBUS_MACROS_H_
#define CDBUS_MACROS_H_

#if defined(__STDC__)
#   define STD_C_C89
#   if defined(__STDC_VERSION__)
#       define STD_C_C90
#   endif
#   if (__STDC_VERSION__ >= 199409L)
#       define STD_C_C94
#   endif
#   if (__STDC_VERSION__ >= 199901L)
#       define STD_C_C99
#   endif
#   if (__STDC_VERSION__ >= 201112L)
#       define STD_C_C11
#   endif
#endif

#ifdef __cplusplus
#   define CDBUS_BEGIN_DECLS extern "C" {
#   define CDBUS_END_DECLS }
#else
#   define CDBUS_BEGIN_DECLS
#   define CDBUS_END_DECLS
#endif

#if defined(_WIN32)
#   if defined(CDBUS_STATIC_BUILD)
#       define CDBUS_EXPORT
#   elif defined(CDBUS_EXPORT_NAMES)
#       define CDBUS_EXPORT __declspec(dllexport)
#   else
#       define CDBUS_EXPORT __declspec(dllimport)
#   endif
#else
#   define CDBUS_EXPORT extern
#endif

#define CDBUS_XSTR(s)   CDBUS_STR(s)
#define CDBUS_STR(s)    #s

#ifndef CDBUS_MAJOR_VERSION
#define CDBUS_MAJOR_VERSION 1
#endif

#ifndef CDBUS_MINOR_VERSION
#define CDBUS_MINOR_VERSION 0
#endif

#ifndef CDBUS_RELEASE_VERSION
#define CDBUS_RELEASE_VERSION 0
#endif

#define CDBUS_VERSION_STRING \
        CDBUS_XSTR(CDBUS_MAJOR_VERSION)"." \
        CDBUS_XSTR(CDBUS_MINOR_VERSION)"." \
        CDBUS_XSTR(CDBUS_RELEASE_VERSION)

#define CDBUS_VERSION ((CDBUS_MAJOR_VERSION << 16) | \
                        (CDBUS_MINOR_VERSION << 8) | \
                        (CDBUS_RELEASE_VERSION))

#endif /* Guard for CDBUS_MACROS_H_ */
