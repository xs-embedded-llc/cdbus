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
 * @file           dbus-watch-ctrl.c        
 * @author         Glenn Schmottlach
 * @brief          Definition of D-Bus related control functions.
 *******************************************************************************
 */
#include <assert.h>
#include "dbus-watch-ctrl.h"
#include "trace.h"
#include "alloc.h"
#include "connection-priv.h"
#include "internal.h"

static cdbus_Bool
cdbus_dbusWatcherHandler
    (
    cdbus_Watch*    w,
    cdbus_UInt32    rcvEvents,
    void *          data
    )
{
    dbus_bool_t handled = TRUE;
    DBusWatch* dbusWatch = (DBusWatch*)data;
    if ( (NULL != w) && (NULL != dbusWatch) )
    {
        handled = dbus_watch_handle(dbusWatch, rcvEvents);
    }

    /* If the callback indicates an OOM condition then ... */
    if ( !handled )
    {
        CDBUS_TRACE((CDBUS_TRC_WARN,
            "D-Bus watch handler returned that needs more memory"));
    }

    return handled;
}


static void
cdbus_dbusFreeWatch
    (
    void*   data
    )
{
    cdbus_Watch* watch = (cdbus_Watch*)data;
    if ( NULL != watch )
    {
        CDBUS_TRACE((CDBUS_TRC_INFO,
            "Unreferencing watch associate with D-Bus watch"));
        cdbus_watchUnref(watch);
    }
}


dbus_bool_t
cdbus_watchAddHandler
    (
    DBusWatch*  dbusWatch,
    void*       data
    )
{
    dbus_bool_t added = FALSE;
    cdbus_Watch* w = NULL;
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    cdbus_Connection* conn = (cdbus_Connection*)data;
    cdbus_Int32 fd;

    assert( NULL != conn );

    if ( NULL != dbusWatch )
    {
#ifdef __linux__
        fd = dbus_watch_get_unix_fd(dbusWatch);
#else
        fd = dbus_watch_get_socket(dbusWatch);
#endif
        w = cdbus_watchNew(conn->dispatcher, fd,
                        dbus_watch_get_flags(dbusWatch),
                        cdbus_dbusWatcherHandler, dbusWatch);
        if ( NULL != w )
        {
            dbus_watch_set_data(dbusWatch, w, cdbus_dbusFreeWatch);

            rc = cdbus_dispatcherAddWatch(conn->dispatcher, w);

            if ( !CDBUS_SUCCEEDED(rc) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to add watch (0x%02x)", rc));
            }
            else
            {
                rc = cdbus_watchEnable(w,
                                    dbus_watch_get_enabled(dbusWatch));
                if ( CDBUS_SUCCEEDED(rc) )
                {
                    added = TRUE;
                }
                else
                {
                    CDBUS_TRACE((CDBUS_TRC_ERROR,
                            "Failed to enable watch (0x%02x)", rc));

                    /* Do best effort to remove the watch we just added */
                    rc = cdbus_dispatcherRemoveWatch(conn->dispatcher, w);
                    if( CDBUS_FAILED(rc) )
                    {
                        CDBUS_TRACE((CDBUS_TRC_ERROR,
                            "Failed removing watch from dispatcher", rc));
                    }
                }

                /*
                 * On error enabling the watch the cdbus_dbusFreeWatch
                 * function *should* be called and unref the cdbus watch
                 * and thus free it up.
                 */
            }
        }
    }
    return added;
}


void
cdbus_watchRemoveHandler
    (
    DBusWatch*  dbusWatch,
    void*       data
    )
{
    cdbus_Watch* w = NULL;
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    cdbus_Connection* conn = (cdbus_Connection*)data;

    assert( NULL != conn );

    if ( NULL != dbusWatch )
    {
        w = dbus_watch_get_data(dbusWatch);
        if ( NULL != w )
        {
            rc = cdbus_watchEnable(w, CDBUS_FALSE);
            if ( CDBUS_FAILED(rc) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR,
                    "Failed to disable the watch (0x%02x)", rc));
            }

            /* Remove the watch from the dispatcher */
            rc = cdbus_dispatcherRemoveWatch(conn->dispatcher, w);
            if( CDBUS_FAILED(rc) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR,
                    "Failed removing watch from dispatcher", rc));
            }

            /* When the D-Bus watch is destroyed it will also
             * unreference our watch proxy. No need to explicitly
             * do it here.
             */
        }
    }
}


void
cdbus_watchToggleHandler
    (
    DBusWatch*  dbusWatch,
    void*       data
    )
{
    cdbus_Connection* conn = (cdbus_Connection*)data;
    cdbus_Watch* w = NULL;
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;

    if ( (NULL != dbusWatch) && (NULL != conn) )
    {
        w = (cdbus_Watch*)dbus_watch_get_data(dbusWatch);
        if ( NULL == w )
        {
            CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to retrieve CDBUS watch"));
        }
        else
        {
            rc = cdbus_watchEnable(w, dbus_watch_get_enabled(dbusWatch));
            if ( CDBUS_FAILED(rc) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR,
                    "Failed to enable watch (0x%02x)", rc));
            }
        }
    }
}



