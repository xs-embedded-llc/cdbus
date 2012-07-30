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
 * @file           introspect-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of the introspectable interface.
 *******************************************************************************
 */
#include "dbus/dbus.h"
#include "cdbus/object.h"
#include "cdbus/introspect.h"
#include "cdbus/stringbuffer.h"
#include "cdbus/connection.h"
#include "trace.h"
#include "internal.h"

static const cdbus_DbusIntrospectArgs cdbus_introspectArg[] =
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
                    dbus_message_iter_init(replyMsg, &iter);
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


