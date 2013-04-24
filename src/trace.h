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
 * @file           trace.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of library trace/debug routines.
 *===========================================================================
 */

#ifndef CDBUS_TRACE_H_
#define CDBUS_TRACE_H_

#include "cdbus/cdbus.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct DBusMessage;


int cdbus_traceIsEnabled(unsigned level, ...);
void cdbus_tracePrintPrefix(int isEnabled, const char* file, const char* funcName, unsigned line);
void cdbus_trace(cdbus_UInt32 level, const cdbus_Char* fmt, ...);
void cdbus_traceMessage(cdbus_UInt32 level, struct DBusMessage* msg);

/*
 * CDBUS_TRACE((LVL, FMT, ...))
 *
 * C89 compatible trace routine. Calls to this macro/function should
 * be written as follows:
 *
 *          CDBUS_TRACE((CDBUS_TRC_DEBUG, "This value is %d", 1));
 */
#ifdef TRACE
    #include <stdio.h>

    #ifdef linux
        #include <libgen.h>
        #define CDBUS_BASENAME(X)   basename(X)
    #else
        #define CDBUS_BASENAME(X)   X
    #endif

#if (__STDC_VERSION__ >= 199901L)
    #define CDBUS_TRACE(X) \
        do { cdbus_tracePrintPrefix(cdbus_traceIsEnabled X, CDBUS_BASENAME(__FILE__), __FUNCTION__, __LINE__); \
        cdbus_trace X; } while ( 0 )
#else
    #define CDBUS_TRACE(X) \
        do { cdbus_tracePrintPrefix(cdbus_traceIsEnabled X, CDBUS_BASENAME(__FILE__), 0, __LINE__); \
        cdbus_trace X; } while ( 0 )
#endif
#else
    #define CDBUS_TRACE(X) do { if ( 0 ) cdbus_trace X; } while ( 0 )
#endif


CDBUS_END_DECLS

#endif /* Guard for CDBUS_TRACE_H_ */
