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
 * @file           trace.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of diagnostic/debug trace routines.
 *===========================================================================
 */
#include <stdio.h>
#include <stdarg.h>
#include <dbus/dbus.h>
#include "trace.h"

static volatile cdbus_UInt32 gsTraceMask = CDBUS_TRC_ALL;

int
cdbus_traceIsEnabled
    (
    unsigned    level,
    ...
    )
{
    return (level & gsTraceMask) != 0;
}


void
cdbus_tracePrintPrefix
    (
    int         isEnabled,
    const char* file,
    const char* funcName,
    unsigned    line
    )
{
    if ( isEnabled )
    {
        if ( NULL != funcName )
        {
            fprintf(stderr, "%s:%s(%u) ", file, funcName, line);
        }
        else
        {
            fprintf(stderr, "%s(%u) ", file, line);
        }
    }
}


void
cdbus_trace
    (
    cdbus_UInt32        level,
    const cdbus_Char*   fmt,
    ...
    )
{
    const char* levelStr = "";
    va_list args;

    if ( level & gsTraceMask )
    {
        switch( level )
        {
            case CDBUS_TRC_OFF:
                break;
            case CDBUS_TRC_FATAL:
                levelStr = "FATAL";
                break;
            case CDBUS_TRC_ERROR:
                levelStr = "ERROR";
                break;
            case CDBUS_TRC_WARN:
                levelStr = "WARN";
                break;
            case CDBUS_TRC_INFO:
                levelStr = "INFO";
                break;
            case CDBUS_TRC_DEBUG:
                levelStr = "DEBUG";
                break;
            case CDBUS_TRC_TRACE:
                levelStr = "TRACE";
                break;
            default:
                break;
        }

        fprintf(stderr, " %s ", levelStr);

        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
    }
}


void
cdbus_traceSetMask
    (
    cdbus_UInt32    mask
    )
{
    gsTraceMask = mask;
}


cdbus_UInt32
cdbus_traceGetMask()
{
    return gsTraceMask;
}


void
cdbus_traceMessage
    (
    cdbus_UInt32        level,
    struct DBusMessage* msg
    )
{
    const cdbus_Char* msgTypeStr ="UNKNOWN";
    cdbus_Int32 msgType = DBUS_MESSAGE_TYPE_INVALID;
    const cdbus_Char* path = NULL;
    const cdbus_Char* intf = NULL;
    const cdbus_Char* name = NULL;
    const cdbus_Char* dest = NULL;
    const cdbus_Char* errName = NULL;


    if ( NULL != msg )
    {
        msgType = dbus_message_get_type(msg);
        msgTypeStr = dbus_message_type_to_string(msgType);
        if ( (DBUS_MESSAGE_TYPE_METHOD_CALL == msgType) ||
            (DBUS_MESSAGE_TYPE_SIGNAL == msgType) )
        {
            path = dbus_message_get_path(msg);
            intf = dbus_message_get_interface(msg);
            name = dbus_message_get_member(msg);
            cdbus_trace(level, "(Ser=%u) [%s] <%s> %s%s%s",
                dbus_message_get_serial(msg),
                msgTypeStr,
                path ? path : "",
                intf ? intf : "",
                intf ? "." : "",
                name ? name : "");
        }
        else if (DBUS_MESSAGE_TYPE_METHOD_RETURN == msgType)
        {
            dest = dbus_message_get_destination(msg);
            cdbus_trace(level, "(RSer=%u) [%s] -> %s",
                        dbus_message_get_reply_serial(msg),
                        msgTypeStr,
                        dest ? dest : "");
        }
        else if (DBUS_MESSAGE_TYPE_ERROR == msgType )
        {
            errName = dbus_message_get_error_name(msg);
            cdbus_trace(level, "(RSer=%u) [%s] %s",
                                    dbus_message_get_reply_serial(msg),
                                    msgTypeStr,
                                    errName ? errName : "");
        }
        else
        {
            cdbus_trace(level, "(Ser=%u) [%s]",
                                            dbus_message_get_serial(msg),
                                            msgTypeStr);
        }
    }
}
