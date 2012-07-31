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
 * @file           connection-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Private implementation of the connection class.
 *******************************************************************************
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

    if ( NULL != userData )
    {
        binder = (cdbus_ObjectConnBinding*)userData;
        cdbus_objectUnref(binder->obj);
        cdbus_connectionUnref(binder->conn);
        cdbus_free(binder);
    }
}


static void
cdbus_freeSetFuncData
    (
    void*   data
    )
{
    cdbus_Connection* conn = (cdbus_Connection*)data;
    if ( NULL != conn )
    {
        cdbus_connectionUnref(conn);
    }
}


cdbus_Connection*
cdbus_connectionNew
    (
    struct cdbus_Dispatcher*    disp
    )
{
    cdbus_Connection* conn = NULL;

    assert( NULL != disp );
    if ( NULL != disp )
    {
        conn = cdbus_calloc(1, sizeof(*conn));
        if ( NULL != conn )
        {
            conn->lock = cdbus_mutexNew(CDBUS_MUTEX_RECURSIVE);
            if ( NULL == conn->lock )
            {
                cdbus_free(conn);
                conn = NULL;
            }
            else
            {
                conn->dispatcher = cdbus_dispatcherRef(disp);
                assert( NULL != conn->dispatcher );
                conn->dbusConn = NULL;
                conn->isPrivate = CDBUS_FALSE;
                cdbus_connectionRef(conn);
                CDBUS_TRACE((CDBUS_TRC_INFO,
                    "Created connection instance (%p)", (void*)conn));
            }
        }
    }

    return conn;
}


void
cdbus_connectionUnref
    (
    cdbus_Connection*   conn
    )
{
    cdbus_Int32 value;
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

            cdbus_dispatcherUnref(conn->dispatcher);
            CDBUS_UNLOCK(conn->lock);
            cdbus_mutexFree(conn->lock);
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


cdbus_HResult
cdbus_connectionOpen
    (
    cdbus_Connection*   conn,
    const cdbus_Char*   address,
    cdbus_Bool          private,
    cdbus_Bool          exitOnDisconnect
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    DBusError dbusError;
    dbus_bool_t status;

    if ( (NULL == conn) || (NULL == address) )
    {
        rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                CDBUS_FAC_CDBUS,
                                CDBUS_EC_INVALID_PARAMETER);
    }
    else
    {
        dbus_error_init(&dbusError);

        CDBUS_LOCK(conn->lock);
        conn->isPrivate = private;
        if ( private )
        {
            conn->dbusConn = dbus_connection_open_private(address, &dbusError);
        }
        else
        {
            conn->dbusConn = dbus_connection_open(address, &dbusError);
        }

        if ( NULL == conn->dbusConn )
        {
            rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE, CDBUS_FAC_DBUS,
                                        CDBUS_EC_CONNECTION_OPEN_FAILURE);
            if ( dbus_error_is_set(&dbusError) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR, "%s : %s", dbusError.name,
                                dbusError.message));
                dbus_error_free(&dbusError);
            }
        }
        else
        {
            dbus_connection_set_exit_on_disconnect(conn->dbusConn,
                                                    exitOnDisconnect);

            status = dbus_connection_set_timeout_functions(conn->dbusConn,
                                                  cdbus_timeoutAddHandler,
                                                  cdbus_timeoutRemoveHandler,
                                                  cdbus_timeoutToggleHandler,
                                                  conn,
                                                  cdbus_freeSetFuncData);
            if ( !status )
            {
                rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE, CDBUS_FAC_DBUS,
                                        CDBUS_EC_ALLOC_FAILURE);
            }
            else
            {
                /* The timeout function handlers have a reference */
                cdbus_connectionRef(conn);

                status = dbus_connection_set_watch_functions(conn->dbusConn,
                                                    cdbus_watchAddHandler,
                                                    cdbus_watchRemoveHandler,
                                                    cdbus_watchToggleHandler,
                                                    conn,
                                                    cdbus_freeSetFuncData);
                if ( !status )
                {
                    rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE, CDBUS_FAC_DBUS,
                                            CDBUS_EC_ALLOC_FAILURE);
                }
                else
                {
                    /* The watch function handlers have a reference */
                    cdbus_connectionRef(conn);
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
            }
        }

        CDBUS_UNLOCK(conn->lock);
    }

    return rc;
}


cdbus_HResult
cdbus_connectionOpenStandard
    (
    cdbus_Connection*   conn,
    DBusBusType         busType,
    cdbus_Bool          private,
    cdbus_Bool          exitOnDisconnect
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    cdbus_Char* addr = NULL;

    switch( busType )
    {
        case DBUS_BUS_SYSTEM:
            addr = getenv("DBUS_SYSTEM_BUS_ADDRESS");
            break;

        case DBUS_BUS_SESSION:
            addr = getenv("DBUS_SESSION_BUS_ADDRESS");
            break;

        case DBUS_BUS_STARTER:
            addr = getenv("DBUS_STARTER_ADDRESS");
            break;

        default:
            rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                    CDBUS_FAC_CDBUS,
                                    CDBUS_EC_INVALID_PARAMETER);
            break;
    }

    if ( NULL != addr )
    {
        rc = cdbus_connectionOpen(conn, addr, private, exitOnDisconnect);
    }

    return rc;
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
            dbus_connection_close(conn->dbusConn);

            /* Once we get the internally generated disconnect
             * message we'll remove the connection from the dispatcher
             * and unreference it.
             */
        }
        else
        {
            /* We no longer want to monitor this connection. The
             * user must still unreference it to free it however.
             */
            rc = cdbus_dispatcherRemoveConnection(conn->dispatcher, conn);
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
cdbus_connectionRegisterObject
    (
    cdbus_Connection*   conn,
    cdbus_Object*       obj
    )
{
    cdbus_Bool isRegistered = CDBUS_FALSE;
    DBusObjectPathVTable vTable = {
                        cdbus_connectionObjectPathUnregisterHandler,
                        cdbus_connectionObjectPathMsgHandler };
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
    }

    return isUnregistered;
}


cdbus_Bool
cdbus_connectionLock
    (
    cdbus_Connection*   conn
    )
{
#ifdef CDBUS_ENABLE_THREAD_SUPPORT
    return cdbus_mutexLock(conn->lock);
#else
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
    return cdbus_mutexUnlock(conn->lock);
#else
    return CDBUS_TRUE;
#endif
}


