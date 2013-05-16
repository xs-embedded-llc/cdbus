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
 * @file           dispatcher-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of the dispatcher API.
 *===========================================================================
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
#include "pointer-pointer-map.h"
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
    CDBUS_EV_UNUSED(EV_A);

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

            /* If the main loop is terminating then ... */
            if ( CDBUS_DISPATCHER_A->exitLoop )
            {
                dbus_connection_flush(conn->dbusConn);
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
            /* Let's schedule another wake up call */
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
    cdbus_ptrPtrMapLock(cdbus_gDispatcherRegistry);
#if EV_MULTIPLICITY
    cdbus_Dispatcher* dispatcher = cdbus_ptrPtrMapGet(cdbus_gDispatcherRegistry, EV_A);
#else
    cdbus_Dispatcher* dispatcher = cdbus_ptrPtrMapGet(cdbus_gDispatcherRegistry, CDBUS_DEFAULT_DISPATCH_LOOP);
#endif
    cdbus_ptrPtrMapUnlock(cdbus_gDispatcherRegistry);

    if ( NULL == CDBUS_DISPATCHER_A )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to lookup dispatcher!"));
    }
    else
    {
        CDBUS_TRACE((CDBUS_TRC_INFO, "Acquiring dispatcher lock."));
        CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
    }
}

static void
cdbus_releaseDispatcherLock
    (
    EV_P
    )
{
    cdbus_ptrPtrMapLock(cdbus_gDispatcherRegistry);
#if EV_MULTIPLICITY
    CDBUS_DISPATCHER_P = cdbus_ptrPtrMapGet(cdbus_gDispatcherRegistry, EV_A);
#else
    CDBUS_DISPATCHER_P = cdbus_ptrPtrMapGet(cdbus_gDispatcherRegistry, CDBUS_DEFAULT_DISPATCH_LOOP);
#endif
    cdbus_ptrPtrMapUnlock(cdbus_gDispatcherRegistry);

    if ( NULL == CDBUS_DISPATCHER_A )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to lookup dispatcher!"));
    }
    else
    {
        CDBUS_TRACE((CDBUS_TRC_INFO, "Releasing dispatcher lock."));
        CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
    }
}


static void
cdbus_runPendingHandlers
    (
    CDBUS_DISPATCHER_P,
    void*               arg
    )
{
    CDBUS_UNUSED(arg);
    ev_invoke_pending(CDBUS_DISPATCHER_LOOP);
}


static void
cdbus_invokePending
    (
    EV_P
    )
{
    cdbus_ptrPtrMapLock(cdbus_gDispatcherRegistry);
#if EV_MULTIPLICITY
    CDBUS_DISPATCHER_P = cdbus_ptrPtrMapGet(cdbus_gDispatcherRegistry, EV_A);
#else
    CDBUS_DISPATCHER_P = cdbus_ptrPtrMapGet(cdbus_gDispatcherRegistry, CDBUS_DEFAULT_DISPATCH_LOOP);
#endif
    cdbus_ptrPtrMapUnlock(cdbus_gDispatcherRegistry);

    if ( NULL == CDBUS_DISPATCHER_A )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to lookup dispatcher!"));
    }
    else
    {
        CDBUS_TRACE((CDBUS_TRC_INFO, "Invoking registered wake-up function."));
        CDBUS_DISPATCHER_A->wakeupFunc(CDBUS_DISPATCHER_A, CDBUS_DISPATCHER_A->wakeupData);

        while ( ev_pending_count(CDBUS_DISPATCHER_LOOP) )
        {
            if ( CDBUS_DISPATCHER_A->wakeupFunc != cdbus_runPendingHandlers )
            {
                CDBUS_SEM_WAIT(CDBUS_DISPATCHER_A->barrier);
                CDBUS_SEM_POST(CDBUS_DISPATCHER_A->barrier);
            }
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

    cdbus_ptrPtrMapLock(cdbus_gDispatcherRegistry);
#if EV_MULTIPLICITY
    /* If no loop specified then assume the default loop */
    if ( NULL == EV_A )
    {
        EV_A = CDBUS_DEFAULT_DISPATCH_LOOP;
    }
    /* See if there is another dispatcher that is already registered
     * with the same event loop.
     */
    CDBUS_DISPATCHER_A = cdbus_ptrPtrMapGet(cdbus_gDispatcherRegistry, EV_A);
#else
    CDBUS_DISPATCHER_A = cdbus_ptrPtrMapGet(cdbus_gDispatcherRegistry, CDBUS_DEFAULT_DISPATCH_LOOP);
#endif
    cdbus_ptrPtrMapUnlock(cdbus_gDispatcherRegistry);

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
            CDBUS_DISPATCHER_A->exitLoop = CDBUS_FALSE;
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
            CDBUS_DISPATCHER_A->finalizerFunc = NULL;
            CDBUS_DISPATCHER_A->finalizerData = NULL;

            LIST_INIT(&CDBUS_DISPATCHER_A->connections);
            LIST_INIT(&CDBUS_DISPATCHER_A->watches);
            LIST_INIT(&CDBUS_DISPATCHER_A->timeouts);
            CDBUS_LOCK_ALLOC(CDBUS_DISPATCHER_A->lock, CDBUS_MUTEX_RECURSIVE);
            CDBUS_DISPATCHER_A->barrier = cdbus_semaphoreNew(0);

            if ( CDBUS_LOCK_IS_NULL(CDBUS_DISPATCHER_A->lock) ||
                (NULL == CDBUS_DISPATCHER_A->barrier) )
            {
                if ( !CDBUS_LOCK_IS_NULL(CDBUS_DISPATCHER_A->lock) )
                {
                    CDBUS_LOCK_FREE(CDBUS_DISPATCHER_A->lock);
                }

                if ( NULL != CDBUS_DISPATCHER_A->barrier )
                {
                    cdbus_semaphoreFree(CDBUS_DISPATCHER_A->barrier);
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
                cdbus_ptrPtrMapLock(cdbus_gDispatcherRegistry);
#if EV_MULTIPLICITY
                status = cdbus_ptrPtrMapAdd(cdbus_gDispatcherRegistry, CDBUS_DISPATCHER_LOOP, CDBUS_DISPATCHER_A);
#else
                status = cdbus_ptrPtrMapAdd(cdbus_gDispatcherRegistry, CDBUS_DEFAULT_DISPATCH_LOOP, CDBUS_DISPATCHER_A);
#endif
                cdbus_ptrPtrMapUnlock(cdbus_gDispatcherRegistry);
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
    void* mapValue = NULL;

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

            /* Remove the loop/dispatcher mapping from the global registry */
            cdbus_ptrPtrMapLock(cdbus_gDispatcherRegistry);
#if EV_MULTIPLICITY
            mapValue = cdbus_ptrPtrMapRemove(cdbus_gDispatcherRegistry,
                                            CDBUS_DISPATCHER_LOOP);
#else
            mapValue = cdbus_ptrPtrMapRemove(cdbus_gDispatcherRegistry,
                                            CDBUS_DEFAULT_DISPATCH_LOOP);
#endif
#ifdef NDEBUG
            CDBUS_UNUSED(mapValue);
#endif
            assert( mapValue == (void*)CDBUS_DISPATCHER_A);

            cdbus_ptrPtrMapUnlock(cdbus_gDispatcherRegistry);

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
                        CDBUS_TRACE((CDBUS_TRC_INFO, "Destroyed libev loop"));
                    }
                }
            }

            /* If there is a hooked defined that needs to know when the
             * Dispatcher is truly destroyed then call it now!
             */
            if ( NULL != CDBUS_DISPATCHER_A->finalizerFunc )
            {
                CDBUS_DISPATCHER_A->finalizerFunc(
                                            CDBUS_DISPATCHER_A->finalizerData);
            }

            CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);

            CDBUS_LOCK_FREE(CDBUS_DISPATCHER_A->lock);
            cdbus_semaphoreFree(CDBUS_DISPATCHER_A->barrier);

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

            /* If we can't add the filter then ... */
            if ( !dbus_connection_add_filter(
                    cdbus_connectionGetDBus(conn),
                    cdbus_connectionFilterHandler, conn, NULL) )
            {
                status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                            CDBUS_FAC_DBUS,
                                            CDBUS_EC_ALLOC_FAILURE);
            }
            else
            {
                status = CDBUS_RESULT_SUCCESS;
            }

            if ( CDBUS_SUCCEEDED(status) )
            {
                /* Add the connection to the list of connections the
                 * dispatcher owns and add a reference to it.
                 */
                LIST_INSERT_HEAD(&CDBUS_DISPATCHER_A->connections,
                                conn, link);
                cdbus_connectionRef(conn);

                dbus_connection_set_dispatch_status_function(
                                cdbus_connectionGetDBus(conn),
                                cdbus_onDispatchStatusChange,
                                conn,
                                NULL /*cdbus_releaseUserData*/);

                dbus_connection_set_wakeup_main_function(
                                cdbus_connectionGetDBus(conn),
                                cdbus_dbusWakeupDispatcher, conn,
                                NULL /*cdbus_releaseUserData*/);

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
                dbus_connection_remove_filter(
                    cdbus_connectionGetDBus(curConn),
                    cdbus_connectionFilterHandler, curConn);
                LIST_REMOVE(curConn, link);

                /* We're no longer handling callbacks for this connection
                 * so we'll NULL out the handlers
                 */
                dbus_connection_set_dispatch_status_function(
                                cdbus_connectionGetDBus(conn),
                                NULL, NULL, NULL);

                dbus_connection_set_wakeup_main_function(
                                cdbus_connectionGetDBus(conn),
                                NULL, NULL, NULL);

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


struct cdbus_Connection*
cdbus_dispatcherGetDbusConnOwner
    (
    CDBUS_DISPATCHER_P,
    DBusConnection*     dbusConn
    )
{
    cdbus_Connection* curConn = NULL;

    if ( NULL != dbusConn )
    {
        CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
        LIST_FOREACH(curConn, &CDBUS_DISPATCHER_A->connections, link)
        {
            if ( curConn->dbusConn == dbusConn )
            {
                break;
            }
        }

        /* If a match is not found then curConn is NULL at
         * the end of this loop
         */
        CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
    }

    return curConn;
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


static cdbus_HResult
cdbus_dispatcherRunImpl
    (
    CDBUS_DISPATCHER_P,
    cdbus_RunOption     runOpt,
    cdbus_Bool          useDispData,
    void*               dispData
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    cdbus_Int32 flag = 0;
    void* oldDispData = NULL;

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
                /* The do/while loop will effectively
                 * implement this run option below.
                 */
                flag = EVRUN_ONCE;
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
            CDBUS_DISPATCHER_A->exitLoop = CDBUS_FALSE;
            do
            {
                CDBUS_LOCK(CDBUS_DISPATCHER_A->lock);
                if ( useDispData )
                {
                    oldDispData = ev_userdata(CDBUS_DISPATCHER_LOOP);
                    ev_set_userdata(CDBUS_DISPATCHER_LOOP_ dispData);
                }
                ev_run(CDBUS_DISPATCHER_LOOP_ flag);
                if ( useDispData )
                {
                    ev_set_userdata(CDBUS_DISPATCHER_LOOP_ oldDispData);
                }
                CDBUS_UNLOCK(CDBUS_DISPATCHER_A->lock);
            }
            while ( !CDBUS_DISPATCHER_A->exitLoop &&
                (CDBUS_RUN_WAIT == runOpt) );

            /* Keeps the state consistent now that the loop
             * has exited.
             */
            CDBUS_DISPATCHER_A->exitLoop = CDBUS_TRUE;
        }
    }

    return rc;
}


cdbus_HResult
cdbus_dispatcherRun
    (
    CDBUS_DISPATCHER_P,
    cdbus_RunOption     runOpt
    )
{
    return cdbus_dispatcherRunImpl(CDBUS_DISPATCHER_A_ runOpt, CDBUS_FALSE,
                                    NULL);
}


cdbus_HResult
cdbus_dispatcherRunWithData
    (
    CDBUS_DISPATCHER_P,
    cdbus_RunOption     runOpt,
    void*               dispData
    )
{
    return cdbus_dispatcherRunImpl(CDBUS_DISPATCHER_A_ runOpt,
                                CDBUS_TRUE, dispData);
}


cdbus_HResult
cdbus_dispatcherStop
    (
    CDBUS_DISPATCHER_P
    )
{
    /* This function purposely does not try to lock
     * the dispatcher so that it is safe to be called
     * from a signal handler. This may make it a little
     * thread-unsafe so it shouldn't be called when another
     * thread might be trying to destroy the dispatcher or
     * start it running again.
     */
    CDBUS_DISPATCHER_A->exitLoop = CDBUS_TRUE;
    /* Force the handler to scan through it's connections
     * and flush everything.
     */
    CDBUS_DISPATCHER_A->dispatchNeeded = CDBUS_TRUE;
    ev_async_send(CDBUS_DISPATCHER_LOOP_ &CDBUS_DISPATCHER_A->asyncWatch);

    return CDBUS_RESULT_SUCCESS;
}


void
cdbus_dispatcherBreak
    (
    CDBUS_DISPATCHER_P
    )
{
    cdbus_dispatcherStop(CDBUS_DISPATCHER_A);
    ev_break(CDBUS_DISPATCHER_LOOP_ EVBREAK_ALL);
}


void
cdbus_dispatcherInvokePending
    (
    CDBUS_DISPATCHER_P
    )
{
    if ( NULL != CDBUS_DISPATCHER_A )
    {
        ev_invoke_pending(CDBUS_DISPATCHER_LOOP);
        CDBUS_SEM_POST(CDBUS_DISPATCHER_A->barrier);
    }
}


void
cdbus_dispatcherSetFinalizer
    (
    CDBUS_DISPATCHER_P,
    cdbus_FinalizerFunc finalizer,
    void*               data
    )
{
    if ( NULL != CDBUS_DISPATCHER_A )
    {
        CDBUS_DISPATCHER_A->finalizerFunc = finalizer;
        CDBUS_DISPATCHER_A->finalizerData = data;
    }
}
