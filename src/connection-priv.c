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
 * @file           connection-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Private implementation of the connection class.
 *===========================================================================
 */
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <cdbus/timeout.h>
#include <cdbus/watch.h>
#include "connection-priv.h"
#include "dispatcher-priv.h"
#include "object-priv.h"
#include "alloc.h"
#include "trace.h"
#include "atomic-ops.h"
#include "dbus-watch-ctrl.h"
#include "dbus-timeout-ctrl.h"
#include "internal.h"
#include "match.h"

#ifndef DBUS_SYSTEM_BUS_DEFAULT_ADDRESS
#define DBUS_SYSTEM_BUS_DEFAULT_ADDRESS "unix:path=/var/run/dbus/system_bus_socket"
#endif

typedef struct cdbus_ObjectConnBinding
{
    cdbus_Object*       obj;
    cdbus_Connection*   conn;
} cdbus_ObjectConnBinding;


static DBusHandlerResult
cdbus_connectionObjectPathMsgHandler
    (
    DBusConnection* dbusConn,
    DBusMessage*    msg,
    void*           userData
    )
{
    DBusHandlerResult result = DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
    cdbus_ObjectConnBinding* binder;

    CDBUS_UNUSED(dbusConn);

    if ( (NULL != msg) && (NULL != userData) )
    {
        cdbus_traceMessage(CDBUS_TRC_TRACE, msg);

        binder = (cdbus_ObjectConnBinding*)userData;
        if ( NULL != binder->obj->handler )
        {
            result = cdbus_objectMessageDispatcher(binder->obj,
                                                    binder->conn, msg);
        }
    }

    return result;
}


static void
cdbus_connectionObjectPathUnregisterHandler
    (
    DBusConnection* dbusConn,
    void*           userData
    )
{
    cdbus_ObjectConnBinding* binder;
    CDBUS_UNUSED(dbusConn);

    if ( NULL != userData )
    {
        binder = (cdbus_ObjectConnBinding*)userData;
        cdbus_objectUnref(binder->obj);
        cdbus_connectionUnref(binder->conn);
        cdbus_free(binder);
    }
}


cdbus_Connection*
cdbus_connectionNew
    (
    struct cdbus_Dispatcher*    disp,
    DBusConnection*             dbusConn,
    cdbus_Bool                  isPrivate
    )
{
    cdbus_Connection* conn = NULL;

    assert( NULL != disp );
    if ( NULL != disp )
    {
        conn = cdbus_calloc(1, sizeof(*conn));
        if ( NULL != conn )
        {
            CDBUS_LOCK_ALLOC(conn->lock, CDBUS_MUTEX_RECURSIVE);
            if ( CDBUS_LOCK_IS_NULL(conn->lock) )
            {
                cdbus_free(conn);
                conn = NULL;
            }
            else
            {
                conn->dispatcher = cdbus_dispatcherRef(disp);
                assert( NULL != conn->dispatcher );
                conn->dbusConn = dbusConn;
                conn->isPrivate = isPrivate;
                LIST_INIT(&conn->matches);
                conn->nextMatch = LIST_END(&conn->matches);
                cdbus_connectionRef(conn);
                CDBUS_TRACE((CDBUS_TRC_INFO,
                    "Created connection instance (%p)", (void*)conn));
            }
        }
    }

    return conn;
}


DBusHandlerResult
cdbus_connectionFilterHandler
    (
    DBusConnection* dbusConn,
    DBusMessage*    msg,
    void*           data
    )
{
    DBusHandlerResult result = DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
    cdbus_Connection* conn = (cdbus_Connection*)data;
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;

    CDBUS_UNUSED(dbusConn);

    if ( NULL != conn )
    {
        /* Only for private connections do you have to look
         * for the Disconnect signal before finalizing the
         * connection.
         */
        if ( conn->isPrivate &&
            dbus_message_is_signal(msg, DBUS_INTERFACE_LOCAL, "Disconnected") &&
            dbus_message_has_path(msg, DBUS_PATH_LOCAL) )
        {
            rc = cdbus_dispatcherRemoveConnection(conn->dispatcher, conn);
            if ( CDBUS_FAILED(rc) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR,
                       "Failed to remove the connection (rc=0x%02X)", rc));
            }
            result = DBUS_HANDLER_RESULT_HANDLED;
        }
        else
        {
            /* Dispatch the message to any registered handlers */
            result = cdbus_connectionDispatchMatches(conn, msg);
        }
    }

    return result;
}


void
cdbus_connectionUnref
    (
    cdbus_Connection*   conn
    )
{
    cdbus_Int32 value;
    cdbus_Match* match;

    assert( NULL != conn );
    if ( NULL != conn )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&conn->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            CDBUS_LOCK(conn->lock);
            if ( NULL != conn->dbusConn )
            {
                if ( dbus_connection_get_is_connected(conn->dbusConn) && conn->isPrivate )
                {
                    /* At this point where the connection is being destroyed the connection
                     * should *not* still be connected. It should've been disconnected
                     * *before* being unreferenced. The D-Bus reference library says you must
                     * wait for a disconnected message before dereferencing the last
                     * connection. We don't have a choice at this point since the dispatcher
                     * has already released it's reference.
                     */
                    dbus_connection_close(conn->dbusConn);
                }

                /* We always unref our D-Bus connection whether it's a private one or
                 * a shared one since we always add a reference when it's created.
                 */
                dbus_connection_unref(conn->dbusConn);
            }

            /* Loop through any matches we have and dispose of them */
            for ( match = LIST_FIRST(&conn->matches);
                match != LIST_END(&conn->matches);
                match = conn->nextMatch )
            {
                conn->nextMatch = LIST_NEXT(match, link);
                cdbus_matchUnref(match);
            }

            cdbus_dispatcherUnref(conn->dispatcher);
            CDBUS_UNLOCK(conn->lock);
            CDBUS_LOCK_FREE(conn->lock);
            cdbus_free(conn);
            CDBUS_TRACE((CDBUS_TRC_INFO,
                    "Destroyed connection instance (%p)", (void*)conn));
        }
    }
}

cdbus_Connection*
cdbus_connectionRef
    (
    cdbus_Connection*   conn
    )
{
    if ( NULL != conn )
    {
        cdbus_atomicAdd(&conn->refCnt, 1);
    }

    return conn;
}


cdbus_Connection*
cdbus_connectionOpen
    (
    cdbus_Dispatcher*   disp,
    const cdbus_Char*   address,
    cdbus_Bool          private,
    cdbus_Bool          exitOnDisconnect
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    DBusError dbusError;
    dbus_bool_t status;
    cdbus_Connection* conn = NULL;
    DBusConnection* dbusConn = NULL;

    if ( (NULL != disp) && (NULL != address) )
    {
        dbus_error_init(&dbusError);

        if ( private )
        {
            dbusConn = dbus_connection_open_private(address, &dbusError);
        }
        else
        {
            dbusConn = dbus_connection_open(address, &dbusError);
        }

        if ( NULL == dbusConn )
        {
            if ( dbus_error_is_set(&dbusError) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR, "%s : %s", dbusError.name,
                                dbusError.message));
                dbus_error_free(&dbusError);
            }
        }
        else
        {
            /* See if there is already a (shared) connection that has the same
             * underlying D-Bus connection.
             */
            conn = cdbus_dispatcherGetDbusConnOwner(disp, dbusConn);
            if ( NULL != conn )
            {
                /* Just add a reference to an (already) existing connection */
                conn = cdbus_connectionRef(conn);
            }
            else
            {
                conn = cdbus_connectionNew(disp, dbusConn, private);
                if ( NULL != conn )
                {

                    dbus_connection_set_exit_on_disconnect(conn->dbusConn,
                                                            exitOnDisconnect);

                    status = dbus_connection_set_timeout_functions(conn->dbusConn,
                                                          cdbus_timeoutAddHandler,
                                                          cdbus_timeoutRemoveHandler,
                                                          cdbus_timeoutToggleHandler,
                                                          conn,
                                                          NULL);
                    if ( !status )
                    {
                        rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE, CDBUS_FAC_DBUS,
                                                CDBUS_EC_ALLOC_FAILURE);
                    }
                    else
                    {
                        status = dbus_connection_set_watch_functions(conn->dbusConn,
                                                            cdbus_watchAddHandler,
                                                            cdbus_watchRemoveHandler,
                                                            cdbus_watchToggleHandler,
                                                            conn,
                                                            NULL);
                        if ( !status )
                        {
                            rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE, CDBUS_FAC_DBUS,
                                                    CDBUS_EC_ALLOC_FAILURE);
                        }
                    }

                    /* If there are no errors up to this point then ... */
                    if ( CDBUS_SUCCEEDED(rc) )
                    {
                        status = dbus_bus_register(conn->dbusConn, &dbusError);
                        if ( !status )
                        {
                            rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE, CDBUS_FAC_DBUS,
                                                    CDBUS_EC_BUS_REG_ERROR);
                            if ( dbus_error_is_set(&dbusError) )
                            {
                                CDBUS_TRACE((CDBUS_TRC_ERROR, "%s : %s", dbusError.name,
                                             dbusError.message));
                                dbus_error_free(&dbusError);
                            }
                        }
                        else
                        {
                            rc = cdbus_dispatcherAddConnection(conn->dispatcher, conn);
                            if ( CDBUS_FAILED(rc) )
                            {
                                CDBUS_TRACE((CDBUS_TRC_ERROR,
                                    "Failed adding connection to the dispatcher (0x%0X)", rc));
                            }
                        }
                    }

                    if ( CDBUS_FAILED(rc) )
                    {
                        if ( conn->isPrivate )
                        {
                            dbus_connection_close(conn->dbusConn);
                        }
                        cdbus_connectionUnref(conn);
                        conn = NULL;
                    }
                }
            }
        }
    }

    return conn;
}


cdbus_Connection*
cdbus_connectionOpenStandard
    (
    cdbus_Dispatcher*   disp,
    DBusBusType         busType,
    cdbus_Bool          private,
    cdbus_Bool          exitOnDisconnect
    )
{
    cdbus_Connection* conn = NULL;
    cdbus_Char* addr = NULL;

    switch( busType )
    {
        case DBUS_BUS_SYSTEM:
            addr = getenv("DBUS_SYSTEM_BUS_ADDRESS");
            if ( NULL == addr )
            {
                /* Try to use the default (hard-coded) address
                 * if the environment variable isn't defined. This
                 * is typically found in the config.h file when the
                 * D-Bus reference library is built.
                 */
                addr = DBUS_SYSTEM_BUS_DEFAULT_ADDRESS;
            }
            break;

        case DBUS_BUS_SESSION:
            addr = getenv("DBUS_SESSION_BUS_ADDRESS");
            break;

        case DBUS_BUS_STARTER:
            addr = getenv("DBUS_STARTER_ADDRESS");
            break;

        default:
            break;
    }

    if ( NULL != addr )
    {
        conn = cdbus_connectionOpen(disp, addr, private, exitOnDisconnect);
    }

    return conn;
}


cdbus_HResult
cdbus_connectionClose
    (
    cdbus_Connection*   conn
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;

    if ( NULL == conn )
    {
        rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                CDBUS_FAC_CDBUS,
                                CDBUS_EC_INVALID_PARAMETER);
    }
    else
    {
        CDBUS_LOCK(conn->lock);

        if ( conn->isPrivate )
        {
            if ( dbus_connection_get_is_connected(conn->dbusConn) )
            {
                dbus_connection_close(conn->dbusConn);
            }

            /* Once we get the internally generated disconnect
             * message we'll remove the connection from the dispatcher
             * and unreference it.
             */
        }
        else
        {
            /* Flush any outstanding messages if connected */
            if ( dbus_connection_get_is_connected(conn->dbusConn) )
            {
                dbus_connection_flush(conn->dbusConn);
            }

            /* We no longer want to monitor this connection. The
             * user must still unreference it, however,  to free it completely.
             */
            rc = cdbus_dispatcherRemoveConnection(conn->dispatcher, conn);
            dbus_connection_unref(conn->dbusConn);
            conn->dbusConn = NULL;
        }

        CDBUS_UNLOCK(conn->lock);
    }

    return rc;
}


DBusConnection*
cdbus_connectionGetDBus
    (
    cdbus_Connection* conn
    )
{
    DBusConnection* dbusConn = NULL;
    CDBUS_LOCK(conn->lock);
    if ( NULL != conn )
    {
        dbusConn = conn->dbusConn;
    }
    CDBUS_UNLOCK(conn->lock);

    return dbusConn;
}


cdbus_Bool
cdbus_connectionGetDescriptor
    (
    cdbus_Connection*   conn,
    cdbus_Descriptor*   descr
    )
{
    cdbus_Bool obtained = CDBUS_FALSE;
    CDBUS_LOCK(conn->lock);
    if ( (NULL != conn) && (NULL != descr) )
    {
        if ( dbus_connection_get_socket(conn->dbusConn, descr) )
        {
            obtained = CDBUS_TRUE;
        }
    }
    CDBUS_UNLOCK(conn->lock);

    return obtained;
}


cdbus_Bool
cdbus_connectionRegisterObject
    (
    cdbus_Connection*   conn,
    cdbus_Object*       obj
    )
{
    cdbus_Bool isRegistered = CDBUS_FALSE;
    DBusObjectPathVTable vTable = {
                        cdbus_connectionObjectPathUnregisterHandler,
                        cdbus_connectionObjectPathMsgHandler, NULL, NULL, NULL, NULL };
    DBusError dbusError;
    cdbus_ObjectConnBinding* binder;

    if ( (NULL != conn) && (NULL != obj) )
    {
        CDBUS_LOCK(conn->lock);

        binder = cdbus_calloc(1, sizeof(*binder));
        if ( NULL == binder )
        {
            cdbus_free(binder);
        }
        else
        {
            binder->obj = obj;
            cdbus_objectRef(obj);
            binder->conn = conn;
            cdbus_connectionRef(conn);

            dbus_error_init(&dbusError);
            if ( dbus_connection_try_register_object_path(conn->dbusConn,
                cdbus_objectGetPath(obj), &vTable, binder, &dbusError) )
            {
                isRegistered = CDBUS_TRUE;
            }
            /* Else we couldn't register the object - cleanup! */
            else
            {
                cdbus_objectUnref(obj);
                cdbus_connectionUnref(conn);
                cdbus_free(binder);
                if ( dbus_error_is_set(&dbusError) )
                {
                    CDBUS_TRACE((CDBUS_TRC_WARN,
                        "Failed to register path: %s", dbusError.message));
                    dbus_error_free(&dbusError);
                }
            }
        }

        CDBUS_UNLOCK(conn->lock);
    }

    return isRegistered;
}


cdbus_Bool
cdbus_connectionUnregisterObject
    (
    cdbus_Connection*   conn,
    const cdbus_Char*   path
    )
{
    cdbus_Bool isUnregistered = CDBUS_FALSE;

    if ( (NULL != conn) && (NULL != path) )
    {
        CDBUS_LOCK(conn->lock);

        if ( dbus_connection_unregister_object_path(conn->dbusConn, path) )
        {
            isUnregistered = CDBUS_TRUE;
        }

        CDBUS_UNLOCK(conn->lock);

        /*
         * The connection is unreferenced when the unregister
         * callback handler is invoked. No need to do it here.
         */
    }

    return isUnregistered;
}


cdbus_Bool
cdbus_connectionSendWithReply
    (
    cdbus_Connection*               conn,
    DBusMessage*                    msg,
    DBusPendingCall**               pending,
    cdbus_Int32                     timeout,
    DBusPendingCallNotifyFunction   notifyFunc,
    void*                           userData,
    DBusFreeFunction                freeUserDataFunc
    )
{
    cdbus_Bool  sent = CDBUS_FALSE;
    DBusPendingCall* localPending = NULL;

    /* Initialize in case of error later */
    if ( NULL != pending )
    {
        *pending = NULL;
    }

    CDBUS_LOCK(conn->lock);
    if ( (NULL != conn ) && (NULL != msg) )
    {
        sent = dbus_connection_send_with_reply(conn->dbusConn, msg, &localPending, timeout);
        if ( sent )
        {
            sent = dbus_pending_call_set_notify(localPending, notifyFunc,
                                                userData, freeUserDataFunc);
            if ( NULL != pending )
            {
                /* Hands off the reference count */
                *pending = localPending;
            }
            else
            {
                dbus_pending_call_unref(localPending);
            }
        }
    }
    CDBUS_UNLOCK(conn->lock);

    return sent;
}


cdbus_Bool
cdbus_connectionLock
    (
    cdbus_Connection*   conn
    )
{
#ifdef CDBUS_ENABLE_THREAD_SUPPORT
    return CDBUS_LOCK(conn->lock);
#else
    CDBUS_UNUSED(conn);
    return CDBUS_TRUE;
#endif
}


cdbus_Bool
cdbus_connectionUnlock
    (
    cdbus_Connection*   conn
    )
{
#ifdef CDBUS_ENABLE_THREAD_SUPPORT
    return CDBUS_UNLOCK(conn->lock);
#else
    CDBUS_UNUSED(conn);
    return CDBUS_TRUE;
#endif
}


cdbus_Handle
cdbus_connectionRegMatchHandler
    (
    cdbus_Connection*               conn,
    cdbus_connectionMatchHandler   handler,
    void*                           userData,
    const cdbus_MatchRule*         rule,
    cdbus_HResult*                  hResult
    )
{
   cdbus_Handle hnd = CDBUS_INVALID_HANDLE;
   cdbus_HResult result = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                               CDBUS_FAC_CDBUS,
                                               CDBUS_EC_INTERNAL);
   cdbus_Match* match = NULL;

   if ( (NULL == conn) || (NULL == handler) )
   {
       result = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                   CDBUS_FAC_CDBUS,
                                   CDBUS_EC_INVALID_PARAMETER);
   }
   else
   {
       CDBUS_LOCK(conn->lock);
       match = cdbus_matchNew(handler, userData, rule);
       if ( NULL == match )
       {
           result = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                      CDBUS_FAC_CDBUS,
                                      CDBUS_EC_ALLOC_FAILURE);
       }

       if ( NULL != match )
       {
           if ( cdbus_matchAddFilter(match, conn) )
           {
               LIST_INSERT_HEAD(&conn->matches, match, link);
               result = CDBUS_RESULT_SUCCESS;
               hnd = match;
           }
           else
           {
               cdbus_matchUnref(match);
               result = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                           CDBUS_FAC_CDBUS,
                                           CDBUS_EC_FILTER_ERROR);
           }
       }
       CDBUS_UNLOCK(conn->lock);
   }


   if ( NULL != hResult )
   {
       *hResult = result;
   }

   return hnd;
}


cdbus_HResult
cdbus_connectionUnregMatchHandler
    (
    cdbus_Connection*   conn,
    cdbus_Handle        regHnd
    )
{
    cdbus_HResult result = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                                CDBUS_FAC_CDBUS,
                                                CDBUS_EC_NOT_FOUND);
    cdbus_Match* match = NULL;

    if ( (NULL == conn) || (CDBUS_INVALID_HANDLE == regHnd) )
    {
        result = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                    CDBUS_FAC_CDBUS,
                                    CDBUS_EC_INVALID_PARAMETER);
    }
    else
    {
        CDBUS_LOCK(conn->lock);

        /* Find the registered signal match and remove it */
        LIST_FOREACH(match, &conn->matches, link)
        {
            if ( (cdbus_Match*)regHnd == match )
            {
                /* If this function was called from a signal handler
                 * callback then we may be trying to unregister the
                 * "next" match in the list. If so bump the "next"
                 * match forward one to skip over the match we're going
                 * to remove.
                 */
                if ( match == conn->nextMatch )
                {
                    conn->nextMatch = LIST_NEXT(match, link);
                }
                LIST_REMOVE(match, link);
                if ( cdbus_matchRemoveFilter(match, conn) )
                {
                    result = CDBUS_RESULT_SUCCESS;
                }
                else
                {
                    result = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                               CDBUS_FAC_CDBUS,
                                               CDBUS_EC_FILTER_ERROR);
                }
                cdbus_matchUnref(match);
                break;
            }
        }
        CDBUS_UNLOCK(conn->lock);
    }

    return result;
}


DBusHandlerResult
cdbus_connectionDispatchMatches
    (
    cdbus_Connection*   conn,
    DBusMessage*        msg
    )
{
    DBusHandlerResult result = DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
    cdbus_Match* match;

    if ( (NULL != conn) && (NULL != msg) )
    {
        CDBUS_LOCK(conn->lock);

        for ( match = LIST_FIRST(&conn->matches);
            match != LIST_END(&conn->matches);
            match = conn->nextMatch )
        {
            conn->nextMatch = LIST_NEXT(match, link);
            if ( cdbus_matchIsMatch(match, msg) )
            {
                /* Add a reference in case the handler tries to
                 * unregister the same handler.
                 */
                cdbus_matchRef(match);
                CDBUS_UNLOCK(conn->lock);
                cdbus_matchDispatch(conn, match, msg);
                CDBUS_LOCK(conn->lock);
                cdbus_matchUnref(match);
                result = DBUS_HANDLER_RESULT_HANDLED;
            }
        }

        /* Reset for the next time this is called */
        conn->nextMatch = LIST_END(&conn->matches);
        CDBUS_UNLOCK(conn->lock);
    }

    return result;
}
