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
 * @file           types.h
 * @author         Glenn Schmottlach
 * @brief          Public types exposed by cdbus.
 *******************************************************************************
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


typedef void (*cdbus_WakeupFunc)(void*);

#ifndef CDBUS_FALSE
#define CDBUS_FALSE     (0U)
#endif

#ifndef CDBUS_TRUE
#define CDBUS_TRUE      (1)
#endif

CDBUS_END_DECLS

#endif /* Guard for CDBUS_TYPES_H_ */
