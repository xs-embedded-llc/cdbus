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
 * @file           trace.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of diagnostic/debug trace routines.
 *******************************************************************************
 */
#include <stdio.h>
#include <stdarg.h>
#include <dbus/dbus.h>
#include "trace.h"

static cdbus_UInt32 gsTraceMask = CDBUS_TRC_ALL;


void
cdbus_trace
    (
    cdbus_UInt32        level,
    const cdbus_Char*   fmt,
    ...
    )
{
    const char* levelStr = "NONE";
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
            CDBUS_TRACE((level, "(Ser=%u) [%s] <%s> %s%s%s",
                dbus_message_get_serial(msg),
                msgTypeStr,
                path ? path : "",
                intf ? intf : "",
                intf ? "." : "",
                name ? name : ""));
        }
        else if (DBUS_MESSAGE_TYPE_METHOD_RETURN == msgType)
        {
            dest = dbus_message_get_destination(msg);
            CDBUS_TRACE((level, "(RSer=%u) [%s] -> %s",
                        dbus_message_get_reply_serial(msg),
                        msgTypeStr,
                        dest ? dest : ""));
        }
        else if (DBUS_MESSAGE_TYPE_ERROR == msgType )
        {
            errName = dbus_message_get_error_name(msg);
            CDBUS_TRACE((level, "(RSer=%u) [%s] %s",
                                    dbus_message_get_reply_serial(msg),
                                    msgTypeStr,
                                    errName ? errName : ""));
        }
        else
        {
            CDBUS_TRACE((level, "(Ser=%u) [%s]",
                                            dbus_message_get_serial(msg),
                                            msgTypeStr));
        }
    }
}
