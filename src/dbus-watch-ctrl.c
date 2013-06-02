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
 * @file           dbus-watch-ctrl.c        
 * @author         Glenn Schmottlach
 * @brief          Definition of D-Bus related control functions.
 *===========================================================================
 */
#include <assert.h>
#include "dbus-watch-ctrl.h"
#include "trace.h"
#include "cdbus/alloc.h"
#include "dispatcher-priv.h"
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
            "Unreferencing watch (%p) associated with D-Bus watch",
            (void*)watch));
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
                CDBUS_TRACE((CDBUS_TRC_INFO, "Added watch (%p) to the dispatcher", w));
                rc = cdbus_watchEnable(w,
                                    dbus_watch_get_enabled(dbusWatch));
                if ( CDBUS_SUCCEEDED(rc) )
                {
                    /* Hold a reference to the connection while D-Bus is
                     * using this watcher.
                     */
                    cdbus_connectionRef(conn);
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
                            "Failed removing watch (%p) from dispatcher (0x%02x)",
                            (void*)w, rc));
                    }
                    /* Failed to add the watcher - unreference the connection */
                    cdbus_connectionUnref(conn);

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
                    "Failed removing watch (%p) from dispatcher (0x%02x)",
                    (void*)w, rc));
            }
            else
            {
                CDBUS_TRACE((CDBUS_TRC_INFO,
                    "Removed watch (%p) from the dispatcher", w));
            }

            /* Since the D-Bus library has disposed of the watcher
             * we'll drop our connection.
             */
            cdbus_connectionUnref(conn);

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



