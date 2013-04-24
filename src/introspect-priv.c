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
 * @file           introspect-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of the introspectable interface.
 *===========================================================================
 */
#include "dbus/dbus.h"
#include "cdbus/object.h"
#include "cdbus/introspect.h"
#include "cdbus/stringbuffer.h"
#include "cdbus/connection.h"
#include "trace.h"
#include "internal.h"

static cdbus_DbusIntrospectArgs cdbus_introspectArg[] =
{
    {"xmlData", "s", CDBUS_XFER_OUT}
};

static const cdbus_DbusIntrospectItem cdbus_introspectMethod[] =
{
    { "Introspect", cdbus_introspectArg, 1},
};


static DBusHandlerResult
cdbus_introspectHandler
(
    struct cdbus_Connection*    conn,
    struct cdbus_Object*        obj,
    DBusMessage*                msg,
    void*                       userData
    )
{
    DBusHandlerResult result = DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
    DBusMessage* replyMsg;
    DBusMessageIter iter;
    CDBUS_UNUSED(userData);
    const cdbus_Char* buf;

    if ( (NULL != conn) && (NULL != obj) && (NULL != msg) )
    {
        if ( dbus_message_has_member(msg, "Introspect") )
        {
            cdbus_StringBuffer* sb = cdbus_objectIntrospect(obj,
                                conn, dbus_message_get_path(msg));
            if ( NULL != sb )
            {
                replyMsg = dbus_message_new_method_return(msg);
                if ( NULL != replyMsg )
                {
                    buf = cdbus_stringBufferRaw(sb);
                    dbus_message_iter_init_append(replyMsg, &iter);
                    if ( dbus_message_iter_append_basic(&iter,
                        DBUS_TYPE_STRING, &buf) )
                    {
                        if ( !dbus_connection_send(cdbus_connectionGetDBus(conn),
                            replyMsg, NULL) )
                        {
                            CDBUS_TRACE((CDBUS_TRC_ERROR,
                                "Failed to send introspection reply"));
                        }
                        else
                        {
                            result = DBUS_HANDLER_RESULT_HANDLED;
                        }
                    }
                    dbus_message_unref(replyMsg);
                }
                cdbus_stringBufferUnref(sb);
            }
        }
    }

    return result;
}


cdbus_Interface*
cdbus_introspectNew()
{
    cdbus_Interface* intf = cdbus_interfaceNew(DBUS_INTERFACE_INTROSPECTABLE,
                                                cdbus_introspectHandler,
                                                NULL);

    if ( NULL != intf )
    {
        /* If we can't register the one method then ... */
        if ( !cdbus_interfaceRegisterMethods(intf,
            cdbus_introspectMethod, 1) )
        {
            /* Failed to register the method - free the resource */
            cdbus_interfaceUnref(intf);
            intf = NULL;
        }
    }

    return intf;
}


