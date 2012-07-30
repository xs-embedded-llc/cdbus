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
 * @file           trace.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of library trace/debug routines.
 *******************************************************************************
 */

#ifndef CDBUS_TRACE_H_
#define CDBUS_TRACE_H_

#include "cdbus/cdbus.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct DBusMessage;

#define CDBUS_TRC_OFF   (0)
#define CDBUS_TRC_FATAL (1 << 5)
#define CDBUS_TRC_ERROR (1 << 4)
#define CDBUS_TRC_WARN  (1 << 3)
#define CDBUS_TRC_INFO  (1 << 2)
#define CDBUS_TRC_DEBUG (1 << 1)
#define CDBUS_TRC_TRACE (1 << 0)
#define CDBUS_TRC_ALL   (~CDBUS_TRC_OFF)

void cdbus_trace(cdbus_UInt32 level, const cdbus_Char* fmt, ...);
void cdbus_traceSetMask(cdbus_UInt32 mask);
cdbus_UInt32 cdbus_traceGetMask();
void cdbus_traceMessage(cdbus_UInt32 level, struct DBusMessage* msg);

/*
 * CDBUS_TRACE((LVL, FMT, ...))
 *
 * C89 compatible trace routine. Calls to this macro/function should
 * be written as follows:
 *
 *          CDBUS_TRACE((CDBUS_TRC_DEBUG, "This value is %d", 1));
 */
#ifdef DEBUG
    #include <stdio.h>

    #ifdef linux
        #include <libgen.h>
        #define CDBUS_BASENAME(X)   basename(X)
    #else
        #define CDBUS_BASENAME(X)   X
    #endif

    #if (__STDC_VERSION__ >= 199901L)
        #define CDBUS_TRACE(X) \
            do { fprintf(stderr, "%s:%s(%u) ", CDBUS_BASENAME(__FILE__), __FUNCTION__, __LINE__); \
            cdbus_trace X; } while ( 0 )
    #else
        #define CDBUS_TRACE(X) \
            do { fprintf(stderr, "%s(%u) ", CDBUS_BASENAME(__FILE__), __LINE__); \
            cdbus_trace X; } while ( 0 )
    #endif
#else
    #define CDBUS_TRACE(X) do { if ( 0 ) cdbus_trace X; } while ( 0 )
#endif


CDBUS_END_DECLS

#endif /* Guard for CDBUS_TRACE_H_ */
