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
 * @file           dispatcher-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of the dispatcher API.
 *******************************************************************************
 */
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <memory.h>
#include "dispatcher-priv.h"
#include "cdbus/dispatcher.h"
#include "dbus/dbus.h"
#include "atomic-ops.h"
#include "alloc.h"
#include "trace.h"
#include "registry.h"
#include "internal.h"


#if EV_MULTIPLICITY
#define CDBUS_DISPATCHER_LOOP CDBUS_DISPATCHER_A->loop
#define CDBUS_DISPATCHER_LOOP_ CDBUS_DISPATCHER_LOOP,
#else
#define CDBUS_DISPATCHER_LOOP
#define CDBUS_DISPATCHER_LOOP_
#endif

#define CDBUS_DEFAULT_DISPATCH_LOOP   ((void*)ev_default_loop(0))

static void
cdbus_releaseUserData
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


static DBusHandlerResult
cdbus_connFilterHandler
    (
    DBusConnection* dbusConn,
    DBusMessage*    msg,
    void*           data
    )
{
    DBusHandlerResult result = DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
    cdbus_Connection* conn = (cdbus_Connection*)data;
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;

    if ( dbus_message_is_signal(msg, DBUS_INTERFACE_LOCAL, "Disconnected") &&
        dbus_message_has_path(msg, DBUS_PATH_LOCAL) )
    {
        if ( NULL != conn )
        {
            /* Only private connections should have filters
             * associated with them.
             */
            assert( conn->isPrivate );
            rc = cdbus_dispatcherRemoveConnection(conn->dispatcher, conn);
            if ( CDBUS_FAILED(rc) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR,
                       "Failed to remove the connection (rc=0x%02X)", rc));
            }
            result = DBUS_HANDLER_RESULT_HANDLED;
        }
    }

    return result;
}


static void
cdbus_onDispatchStatusChange
    (
    DBusConnection*     dbusConn,
    DBusDispatchStatus  newStatus,
    void*               data
    )
{
    CDBUS_UNUSED(dbusConn);
    cdbus_Connection* conn = (cdbus_Connection*)data;

    if ( DBUS_DISPATCH_DATA_REMAINS == newStatus )
    {
        if ( NULL == conn )
        {
            CDBUS_TRACE((CDBUS_TRC_ERROR, "Connection is unset!"));
        }
        else
        {
            /* There are incoming messages that need to be processed. We
             * *cannot* call dbus_connection_dispatch() directly because
             * reentrancy is not allowed in this situation. We'll have to
             * wake up the dispatcher and tell it to dispatch for us.
             */
            conn->dispatcher->dispatchNeeded = CDBUS_TRUE;
            cdbus_dispatcherWakeup(conn->dispatcher);
        }
    }
}


static void
cdbus_asyncCallback
    (
    EV_P_
    ev_async*       w,
    int             rcvEvents
    )
{
    cdbus_Connection* conn;
    cdbus_Connection* nextConn;
    DBusDispatchStatus status;

    /* Used for the side-effect of waking up the event loop */
    CDBUS_UNUSED(rcvEvents);

    cdbus_Dispatcher* CDBUS_DISPATCHER_A = (cdbus_Dispatcher*)w->data;
    if ( CDBUS_DISPATCHER_A->dispatchNeeded )
    {
        /* Let's assume (possibly incorrectly) that all the dispatching
         * is completed here
         */
        CDBUS_DISPATCHER_A->dispatchNeeded = CDBUS_FALSE;

        /* Loop through all the connections */
        for ( conn = LIST_FIRST(&CDBUS_DISPATCHER_A->connections);
            conn != LIST_END(&CDBUS_DISPATCHER_A->connections);
            conn = nextConn )
        {
            nextConn = LIST_NEXT(conn, link);
            status = dbus_connection_get_dispatch_status(cdbus_connectionGetDBus(conn));
            if ( DBUS_DISPATCH_DATA_REMAINS == status )
            {
                status = dbus_connection_dispatch(conn->dbusConn);
            }

            if ( DBUS_DISPATCH_COMPLETE != status )
            {
                /* It looks like we need to do more dispatching
                 * to drain the message queue.
                 */
                CDBUS_DISPATCHER_A->dispatchNeeded = CDBUS_TRUE;
            }
        }

        /* If we processed a connection that did not complete
         * dispatching for some reason then ...
         */
        if ( CDBUS_DISPATCHER_A->dispatchNeeded )
        {
            /* Let's schedule another wakeup call */
            cdbus_dispatcherWakeup(CDBUS_DISPATCHER_A);
        }
    }
}


static void
cdbus_acquireDispatcherLock
    (
    EV_P
    )
{
#if EV_MULTIPLICITY
    cdbus_Dispatcher* dispatcher = cdbus_registryGet(cdbus_gDispatcherRegistry, EV_A);
#else
    cdbus_Dispatcher* dispatcher = cdbus_registryGet(cdbus_gDispatcherRegistry, CDBUS_DEFAULT_DISPATCH_LOOP);
#endif
    if ( NULL == CDBUS_DISPATCHER_A )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to lookup dispatcher!"));
    }
    else
    {
        CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
    }
}

static void
cdbus_releaseDispatcherLock
    (
    EV_P
    )
{
#if EV_MULTIPLICITY
    CDBUS_DISPATCHER_P = cdbus_registryGet(cdbus_gDispatcherRegistry, EV_A);
#else
    CDBUS_DISPATCHER_P = cdbus_registryGet(cdbus_gDispatcherRegistry, CDBUS_DEFAULT_DISPATCH_LOOP);
#endif

    if ( NULL == CDBUS_DISPATCHER_A )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to lookup dispatcher!"));
    }
    else
    {
        CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
    }
}


static void
cdbus_runPendingHandlers
    (
    void* arg
    )
{
    CDBUS_DISPATCHER_P = (cdbus_Dispatcher*)arg;

    assert( NULL != CDBUS_DISPATCHER_A );
    //CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
    ev_invoke_pending(CDBUS_DISPATCHER_LOOP);
    //CDBUS_CV_SIGNAL(CDBUS_DISPATCHER_A->cv);
    //CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
}


static void
cdbus_invokePending
    (
    EV_P
    )
{
#if EV_MULTIPLICITY
    CDBUS_DISPATCHER_P = cdbus_registryGet(cdbus_gDispatcherRegistry, EV_A);
#else
    CDBUS_DISPATCHER_P = cdbus_registryGet(cdbus_gDispatcherRegistry, CDBUS_DEFAULT_DISPATCH_LOOP);
#endif

    if ( NULL == CDBUS_DISPATCHER_A )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to lookup dispatcher!"));
    }
    else
    {
        while ( ev_pending_count(CDBUS_DISPATCHER_LOOP) )
        {
            /* Call the user supplied thread "wake up" function */
            CDBUS_DISPATCHER_A->wakeupFunc(CDBUS_DISPATCHER_A->wakeupData);
            //CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
            //CDBUS_CV_WAIT(CDBUS_DISPATCHER_A->cv, CDBUS_DISPATCHER_A->lock);
            //CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
        }
    }
}


static void
cdbus_dbusWakeupDispatcher
    (
    void*   data
    )
{
    cdbus_Connection* conn = (cdbus_Connection*)(data);
    if ( NULL == conn )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Connection not provided to wake-up dispatcher!"));
    }
    else
    {
        cdbus_dispatcherWakeup(conn->dispatcher);
    }
}


cdbus_Dispatcher*
cdbus_dispatcherNew
    (
    EV_P_
    cdbus_Bool          ownsLoop,
    cdbus_WakeupFunc    wakeupFunc,
    void*               wakeupData
    )
{
    CDBUS_DISPATCHER_P = NULL;
    cdbus_Bool status;

#if EV_MULTIPLICITY
    /* If no loop specified then assume the default loop */
    if ( NULL == EV_A )
    {
        EV_A = CDBUS_DEFAULT_DISPATCH_LOOP;
    }
    /* See if there is another dispatcher that is already registered
     * with the same event loop.
     */
    CDBUS_DISPATCHER_A = cdbus_registryGet(cdbus_gDispatcherRegistry, EV_A);
#else
    CDBUS_DISPATCHER_A = cdbus_registryGet(cdbus_gDispatcherRegistry, CDBUS_DEFAULT_DISPATCH_LOOP);
#endif
    if ( NULL != CDBUS_DISPATCHER_A )
    {
        /* A matching dispatcher already exists so we'll add a reference
         * and return that as the dispatcher.
         */
        cdbus_dispatcherRef(CDBUS_DISPATCHER_A);
    }
    /* Else this is a completely new dispatcher */
    else
    {
        CDBUS_DISPATCHER_A = cdbus_calloc(1, sizeof(*CDBUS_DISPATCHER_A));
        if ( NULL != CDBUS_DISPATCHER_A )
        {
#if EV_MULTIPLICITY
            CDBUS_DISPATCHER_LOOP = EV_A;
#endif
            CDBUS_DISPATCHER_A->ownsLoop = ownsLoop;
            if ( NULL != wakeupFunc )
            {
                CDBUS_DISPATCHER_A->wakeupFunc = wakeupFunc;
                CDBUS_DISPATCHER_A->wakeupData = wakeupData;
            }
            else
            {
                CDBUS_DISPATCHER_A->wakeupFunc = cdbus_runPendingHandlers;
                CDBUS_DISPATCHER_A->wakeupData = CDBUS_DISPATCHER_A;
            }
            LIST_INIT(&CDBUS_DISPATCHER_A->connections);
            LIST_INIT(&CDBUS_DISPATCHER_A->watches);
            LIST_INIT(&CDBUS_DISPATCHER_A->timeouts);
            CDBUS_DISPATCHER_A->lock = cdbus_mutexNew(CDBUS_MUTEX_RECURSIVE);
            CDBUS_DISPATCHER_A->cv = cdbus_condVarNew();
            if ( (NULL == CDBUS_DISPATCHER_A->lock) ||
                (NULL == CDBUS_DISPATCHER_A->cv) )
            {
                if ( NULL != CDBUS_DISPATCHER_A->lock )
                {
                    cdbus_mutexFree(CDBUS_DISPATCHER_A->lock);
                }

                if ( NULL != CDBUS_DISPATCHER_A->cv )
                {
                    cdbus_condVarFree(CDBUS_DISPATCHER_A->cv);
                }
                cdbus_free(CDBUS_DISPATCHER_A);
                CDBUS_DISPATCHER_A = NULL;
            }
            else
            {
                CDBUS_DISPATCHER_A->dispatchNeeded = CDBUS_FALSE;
                ev_async_init(&CDBUS_DISPATCHER_A->asyncWatch, cdbus_asyncCallback);
                CDBUS_DISPATCHER_A->asyncWatch.data = CDBUS_DISPATCHER_A;
                ev_async_start(EV_A_ &CDBUS_DISPATCHER_A->asyncWatch);
                ev_set_invoke_pending_cb(EV_A_ cdbus_invokePending);
                ev_set_loop_release_cb(EV_A_ cdbus_releaseDispatcherLock,
                                        cdbus_acquireDispatcherLock);
                ev_ref(EV_A);
                cdbus_dispatcherRef(CDBUS_DISPATCHER_A);
                CDBUS_TRACE((CDBUS_TRC_INFO,
                    "Created dispatcher instance (%p)", (void*)CDBUS_DISPATCHER_A));
            }

            if ( NULL != CDBUS_DISPATCHER_A )
            {
                /* Register the loop to the dispatcher */
#if EV_MULTIPLICITY
                status = cdbus_registryAdd(cdbus_gDispatcherRegistry, CDBUS_DISPATCHER_LOOP, CDBUS_DISPATCHER_A);
#else
                status = cdbus_registryAdd(cdbus_gDispatcherRegistry, CDBUS_DEFAULT_DISPATCH_LOOP, CDBUS_DISPATCHER_A);
#endif
                if ( !status )
                {
                    CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to add dispatcher to registry!"));
                }
            }
        }
    }

    return CDBUS_DISPATCHER_A;
}


void
cdbus_dispatcherUnref
    (
    CDBUS_DISPATCHER_P
    )
{
    cdbus_Int32 value = 0;
    cdbus_Connection* conn = NULL;
    cdbus_Connection* nextConn = NULL;

    assert( NULL != CDBUS_DISPATCHER_A );
    if ( NULL != CDBUS_DISPATCHER_A )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&CDBUS_DISPATCHER_A->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);

            /* Destroy all the connections */
            for ( conn = LIST_FIRST(&CDBUS_DISPATCHER_A->connections);
                conn != LIST_END(&CDBUS_DISPATCHER_A->connections);
                conn = nextConn )
            {
                nextConn = LIST_NEXT(conn, link);
                cdbus_dispatcherRemoveConnection(CDBUS_DISPATCHER_A, conn);
            }

#if EV_MULTIPLICITY
            /* Destroy the dispatcher itself */
            if ( NULL != CDBUS_DISPATCHER_LOOP )
#else
            if ( CDBUS_TRUE )
#endif
            {
                /* Stop the async watcher used to wake up the main loop in
                 * multi-threaded scenarios.
                 */
                ev_async_stop(CDBUS_DISPATCHER_LOOP_ &dispatcher->asyncWatch);
                /* Drop our reference to the loop */
                ev_unref(CDBUS_DISPATCHER_LOOP);

                /* If we own and control this main loop then ... */
                if ( CDBUS_DISPATCHER_A->ownsLoop )
                {
                    /* Destroy the event loop */
                    ev_break(CDBUS_DISPATCHER_LOOP_ EVBREAK_ALL);
                    if ( !ev_is_default_loop(CDBUS_DISPATCHER_LOOP) )
                    {
                        ev_loop_destroy(CDBUS_DISPATCHER_LOOP);
                    }
                }
#if EV_MULTIPLICITY
                CDBUS_DISPATCHER_LOOP = NULL;
#endif
            }

            CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);

            cdbus_mutexFree(CDBUS_DISPATCHER_A->lock);
            cdbus_condVarFree(CDBUS_DISPATCHER_A->cv);

            /* Free the dispatcher itself */
            cdbus_free(CDBUS_DISPATCHER_A);
            CDBUS_TRACE((CDBUS_TRC_INFO,
                 "Destroyed dispatcher instance (%p)", (void*)CDBUS_DISPATCHER_A));
        }
    }
}


cdbus_Dispatcher*
cdbus_dispatcherRef
    (
    CDBUS_DISPATCHER_P
    )
{
    if ( NULL != CDBUS_DISPATCHER_A )
    {
        cdbus_atomicAdd(&CDBUS_DISPATCHER_A->refCnt, 1);
    }

    return CDBUS_DISPATCHER_A;
}

cdbus_HResult
cdbus_dispatcherAddConnection
    (
    CDBUS_DISPATCHER_P,
    struct cdbus_Connection*    conn
    )
{
    cdbus_Connection*   curConn = NULL;
    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                            CDBUS_FAC_CDBUS,
                            CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != conn) && (NULL != CDBUS_DISPATCHER_A) )
    {
        CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);

        /* Only add the connection if it doesn't already exist in the list */
        LIST_FOREACH(curConn, &CDBUS_DISPATCHER_A->connections, link)
        {
            if ( curConn == conn )
            {
                break;
            }
        }

        /* If the connection is already in the list of connections then ... */
        if ( curConn != LIST_END(&CDBUS_DISPATCHER_A->connections) )
        {
            status = CDBUS_RESULT_SUCCESS;
        }
        /* Else this connection doesn't exist in the list */
        else
        {
            /* Every callback will own a reference to the connection so
             * that it's not inadvertently freed before these callbacks
             * can be called. Each callback has a "free" function that is
             * used to unreference the connection.
             */

            /* We only filter private connections looking for internally
             * generated disconnect signals
             */
            if ( !conn->isPrivate )
            {
                status = CDBUS_RESULT_SUCCESS;
            }
            else
            {
                /* If we can't add the filter then ... */
                if ( !dbus_connection_add_filter(
                        cdbus_connectionGetDBus(conn),
                        cdbus_connFilterHandler, conn,
                        cdbus_releaseUserData) )
                {
                    status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                                CDBUS_FAC_DBUS,
                                                CDBUS_EC_ALLOC_FAILURE);
                }
                else
                {
                    /* Filter created so add a reference to the connection */
                    cdbus_connectionRef(conn);
                }
            }

            if ( CDBUS_SUCCEEDED(status) )
            {
                /* Add the connection to the list of connections the
                 * dispatcher owns and add a reference to it.
                 */
                LIST_INSERT_HEAD(&CDBUS_DISPATCHER_A->connections,
                                conn, link);
                cdbus_connectionRef(conn);


                cdbus_connectionRef(conn);
                dbus_connection_set_dispatch_status_function(
                                cdbus_connectionGetDBus(conn),
                                cdbus_onDispatchStatusChange,
                                conn,
                                cdbus_releaseUserData);

                cdbus_connectionRef(conn);
                dbus_connection_set_wakeup_main_function(
                                cdbus_connectionGetDBus(conn),
                                cdbus_dbusWakeupDispatcher, conn,
                                cdbus_releaseUserData);

                if ( DBUS_DISPATCH_DATA_REMAINS ==
                    dbus_connection_get_dispatch_status(cdbus_connectionGetDBus(conn)) )
                {
                    CDBUS_DISPATCHER_A->dispatchNeeded = CDBUS_TRUE;
                }
                cdbus_dispatcherWakeup(CDBUS_DISPATCHER_A);
            }
        }
        CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
    }

    return status;
}


cdbus_HResult
cdbus_dispatcherRemoveConnection
    (
    CDBUS_DISPATCHER_P,
    struct cdbus_Connection*    conn
    )
{
    cdbus_Connection* curConn = NULL;

    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != conn) && (NULL != CDBUS_DISPATCHER_A) )
    {
        CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
        LIST_FOREACH(curConn, &CDBUS_DISPATCHER_A->connections, link)
        {
            if ( curConn == conn )
            {
                if ( curConn->isPrivate )
                {
                    dbus_connection_remove_filter(
                        cdbus_connectionGetDBus(curConn),
                        cdbus_connFilterHandler, curConn);
                }
                LIST_REMOVE(curConn, link);
                cdbus_connectionUnref(conn);
                status = CDBUS_RESULT_SUCCESS;
                break;
            }
        }

        /* If a matching connection was not found then ... */
        if ( LIST_END(&CDBUS_DISPATCHER_A->connections) == curConn )
        {
            status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_NOT_FOUND);
        }
        else
        {
            /* If we're running multi-threaded then wake up the event loop
             * so it notices that the connection has been removed.
             */
            cdbus_dispatcherWakeup(CDBUS_DISPATCHER_A);
        }
        CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
    }

    return status;
}


cdbus_HResult
cdbus_dispatcherAddWatch
    (
    CDBUS_DISPATCHER_P,
    struct cdbus_Watch* watch
    )
{
    cdbus_Watch* curWatch = NULL;
    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                            CDBUS_FAC_CDBUS,
                            CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != watch) && (NULL != CDBUS_DISPATCHER_A) )
    {
        CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
        /* Only add the watch if it doesn't already exist in the list */
        LIST_FOREACH(curWatch, &CDBUS_DISPATCHER_A->watches, link)
        {
            if ( curWatch == watch )
            {
                break;
            }
        }

        /* If the connection isn't already in the list of connections then ... */
        if ( curWatch == LIST_END(&CDBUS_DISPATCHER_A->watches) )
        {
            LIST_INSERT_HEAD(&CDBUS_DISPATCHER_A->watches, watch, link);

            /* This dispatcher now references it too */
            cdbus_watchRef(watch);

            /* Start the watch */
            ev_io_start(CDBUS_DISPATCHER_LOOP_ &watch->ioWatcher);
            cdbus_dispatcherWakeup(CDBUS_DISPATCHER_A);
        }
        CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);

        status = CDBUS_RESULT_SUCCESS;
    }

    return status;
}


cdbus_HResult
cdbus_dispatcherRemoveWatch
    (
    CDBUS_DISPATCHER_P,
    struct cdbus_Watch* watch
    )
{
    cdbus_Watch* curWatch = NULL;

    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != watch) && (NULL != CDBUS_DISPATCHER_A) )
    {
        CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
        LIST_FOREACH(curWatch, &CDBUS_DISPATCHER_A->watches, link)
        {
            if ( curWatch == watch )
            {
                LIST_REMOVE(curWatch, link);
                /* Start the watch */
                ev_io_stop(CDBUS_DISPATCHER_LOOP_ &watch->ioWatcher);
                cdbus_watchUnref(watch);
                status = CDBUS_RESULT_SUCCESS;
                break;
            }
        }

        /* If a matching watch was not found then ... */
        if ( LIST_END(&CDBUS_DISPATCHER_A->watches) == curWatch )
        {
            status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_NOT_FOUND);
        }
        else
        {
            /* If we're running multi-threaded then wake up the event loop
             * so it notices that the watch has been removed.
             */
            cdbus_dispatcherWakeup(CDBUS_DISPATCHER_A);
        }
        CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
    }

    return status;
}


cdbus_HResult
cdbus_dispatcherAddTimeout
    (
    CDBUS_DISPATCHER_P,
    struct cdbus_Timeout*   timeout
    )
{
    cdbus_Timeout* curTimeout = NULL;
    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                            CDBUS_FAC_CDBUS,
                            CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != timeout) && (NULL != CDBUS_DISPATCHER_A) )
    {
        CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
        /* Only add the watch if it doesn't already exist in the list */
        LIST_FOREACH(curTimeout, &CDBUS_DISPATCHER_A->timeouts, link)
        {
            if ( curTimeout == timeout )
            {
                break;
            }
        }

        /* If the connection isn't already in the list of connections then ... */
        if ( curTimeout == LIST_END(&CDBUS_DISPATCHER_A->timeouts) )
        {
            LIST_INSERT_HEAD(&CDBUS_DISPATCHER_A->timeouts, timeout, link);
            /* The dispatcher now holds a reference to the timeout */
            cdbus_timeoutRef(timeout);

            ev_timer_again(CDBUS_DISPATCHER_LOOP_ &timeout->timerWatcher);
            cdbus_dispatcherWakeup(CDBUS_DISPATCHER_A);
        }
        CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);

        status = CDBUS_RESULT_SUCCESS;
    }

    return status;
}


cdbus_HResult
cdbus_dispatcherRemoveTimeout
    (
    CDBUS_DISPATCHER_P,
    struct cdbus_Timeout*   timeout
    )
{
    cdbus_Timeout* curTimeout = NULL;

    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != timeout) && (NULL != CDBUS_DISPATCHER_A) )
    {
        CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
        LIST_FOREACH(curTimeout, &CDBUS_DISPATCHER_A->timeouts, link)
        {
            if ( curTimeout == timeout )
            {
                LIST_REMOVE(curTimeout, link);
                ev_timer_stop(CDBUS_DISPATCHER_LOOP_ &timeout->timerWatcher);
                cdbus_timeoutUnref(timeout);
                status = CDBUS_RESULT_SUCCESS;
                break;
            }
        }

        /* If a matching timeout was not found then ... */
        if ( LIST_END(&CDBUS_DISPATCHER_A->timeouts) == curTimeout )
        {
            status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_NOT_FOUND);
        }
        else
        {
            /* If we're running multi-threaded then wake up the event loop
             * so it notices that the timeout has been removed.
             */
            cdbus_dispatcherWakeup(CDBUS_DISPATCHER_A);
        }
        CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
    }

    return status;
}


void cdbus_dispatcherWakeup
    (
    CDBUS_DISPATCHER_P
    )
{
    if ( NULL != CDBUS_DISPATCHER_A )
    {
        CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
        ev_async_send(CDBUS_DISPATCHER_LOOP_ &CDBUS_DISPATCHER_A->asyncWatch);
        CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
    }
}


cdbus_HResult
cdbus_dispatcherRun
    (
    CDBUS_DISPATCHER_P,
    cdbus_RunOption     runOpt
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    cdbus_Int32 flag = 0;

    if ( NULL == CDBUS_DISPATCHER_A )
    {
        rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                CDBUS_FAC_CDBUS,
                                CDBUS_EC_INVALID_PARAMETER);
    }
    else
    {
        switch ( runOpt )
        {
            case CDBUS_RUN_WAIT:
                flag = 0;
                break;

            case CDBUS_RUN_NO_WAIT:
                flag = EVRUN_NOWAIT;
                break;

            case CDBUS_RUN_ONCE:
                flag = EVRUN_ONCE;
                break;

            default:
                rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_INVALID_PARAMETER);
                CDBUS_TRACE((CDBUS_TRC_ERROR, "Unknown run option (%d)"));
                break;
        }

        if ( CDBUS_SUCCEEDED(rc) )
        {
            CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
            ev_run(CDBUS_DISPATCHER_LOOP_ flag);
            CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
        }
    }

    return rc;
}


cdbus_HResult
cdbus_dispatcherBreak
    (
    CDBUS_DISPATCHER_P,
    cdbus_BreakOption   opt
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    cdbus_Int32 how = EVBREAK_ALL;

    if ( NULL == CDBUS_DISPATCHER_A )
    {
        rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                CDBUS_FAC_CDBUS,
                                CDBUS_EC_INVALID_PARAMETER);
    }
    else
    {
        if ( opt == CDBUS_BREAK_ALL )
        {
            how = EVBREAK_ALL;
        }
        else if ( opt == CDBUS_BREAK_ONE )
        {
            how = EVBREAK_ONE;
        }
        else
        {
            CDBUS_TRACE((CDBUS_TRC_ERROR, "Unknown break option (%d)"));
            rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                    CDBUS_FAC_CDBUS,
                                    CDBUS_EC_INVALID_PARAMETER);
        }


        if ( CDBUS_SUCCEEDED(rc) )
        {
            CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
            ev_break(CDBUS_DISPATCHER_LOOP_ how);
            CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
        }
    }

    return rc;
}

