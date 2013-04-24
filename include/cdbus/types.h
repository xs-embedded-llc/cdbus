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
 * @file           types.h
 * @author         Glenn Schmottlach
 * @brief          Public types exposed by cdbus.
 *===========================================================================
 */

#ifndef CDBUS_TYPES_H_
#define CDBUS_TYPES_H_

#include "cdbus/macros.h"

CDBUS_BEGIN_DECLS

#if (__STDC_VERSION__ >= 199901L)
#include <stdint.h>

typedef unsigned char       cdbus_UChar;
typedef char                cdbus_Char;
typedef uint8_t             cdbus_UInt8;
typedef int8_t              cdbus_Int8;
typedef int16_t             cdbus_Int16;
typedef uint16_t            cdbus_UInt16;
typedef int32_t             cdbus_Int32;
typedef uint32_t            cdbus_UInt32;
typedef int64_t             cdbus_Int64;
typedef uint64_t            cdbus_UInt64;


#else

typedef unsigned char       cdbus_UChar;
typedef char                cdbus_Char;
typedef unsigned char       cdbus_UInt8;
typedef char                cdbus_Int8;
typedef short               cdbus_Int16;
typedef unsigned short      cdbus_UInt16;
typedef int                 cdbus_Int32;
typedef unsigned int        cdbus_UInt32;
typedef long long int       cdbus_Int64;
typedef unsigned long int   cdbus_UInt64;

#endif

typedef cdbus_UChar         cdbus_Bool;
typedef cdbus_UInt32        cdbus_HResult;
typedef cdbus_Int32         cdbus_Descriptor;
typedef void*               cdbus_Handle;

#define CDBUS_INVALID_HANDLE    ((void*)0)

typedef struct cdbus_Atomic
{
    volatile cdbus_Int32 value;
} cdbus_Atomic;


#ifndef CDBUS_FALSE
#define CDBUS_FALSE     (0U)
#endif

#ifndef CDBUS_TRUE
#define CDBUS_TRUE      (1)
#endif

CDBUS_END_DECLS

#endif /* Guard for CDBUS_TYPES_H_ */
