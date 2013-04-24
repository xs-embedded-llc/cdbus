/*===========================================================================
 * 
 * Project         cdbus
 *
 * Released under the MIT License (MIT)
 * Copyright (c) 2013 XS-Embedded LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
 * NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *===========================================================================
 *===========================================================================
 * @file           macros.h        
 * @author         Glenn Schmottlach
 * @brief          Definition of common macros
 *===========================================================================
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
