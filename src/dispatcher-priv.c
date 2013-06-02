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
#include "cdbus/atomic-ops.h"
#include "cdbus/alloc.h"
#include "cdbus/mainloop.h"
#include "dbus/dbus.h"
#include "trace.h"
#include "pointer-pointer-map.h"
#include "internal.h"
#include "pipe.h"


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
cdbus_wakeupHandler
    (
    cdbus_MainLoopWatch*    watch,
    cdbus_UInt32            flags,
    void*                   userData
    )
{
    cdbus_Connection* conn;
    cdbus_Connection* nextConn;
    DBusDispatchStatus status;
    cdbus_Char         pipeBuf;

    cdbus_Dispatcher* dispatcher = (cdbus_Dispatcher*)userData;
    assert( NULL != dispatcher );

    CDBUS_UNUSED(watch);
    CDBUS_UNUSED(flags);

    /* Drain the wakeuup pipe of any data */
    while ( sizeof(pipeBuf) == cdbus_pipeRead(dispatcher->wakeupPipe, &pipeBuf,
        sizeof(pipeBuf)) );
    dispatcher->wakeupTrigger = CDBUS_FALSE;

    if ( dispatcher->dispatchNeeded )
    {
        /* Let's assume (possibly incorrectly) that all the dispatching
         * is completed here
         */
        dispatcher->dispatchNeeded = CDBUS_FALSE;

        /* Loop through all the connections */
        for ( conn = LIST_FIRST(&dispatcher->connections);
            conn != LIST_END(&dispatcher->connections);
            conn = nextConn )
        {
            nextConn = LIST_NEXT(conn, link);
            status = dbus_connection_get_dispatch_status(
                                            cdbus_connectionGetDBus(conn));
            if ( DBUS_DISPATCH_DATA_REMAINS == status )
            {
                status = dbus_connection_dispatch(conn->dbusConn);
            }

            /* If the main loop is terminating then ... */
            if ( dispatcher->exitLoop )
            {
                dbus_connection_flush(conn->dbusConn);
            }

            if ( DBUS_DISPATCH_COMPLETE != status )
            {
                /* It looks like we need to do more dispatching
                 * to drain the message queue.
                 */
                dispatcher->dispatchNeeded = CDBUS_TRUE;
            }
        }

        /* If we processed a connection that did not complete
         * dispatching for some reason then ...
         */
        if ( dispatcher->dispatchNeeded )
        {
            /* Let's schedule another wake up call */
            cdbus_dispatcherWakeup(dispatcher);
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
        CDBUS_TRACE((CDBUS_TRC_ERROR,
                    "Connection not provided to wake-up dispatcher!"));
    }
    else
    {
        cdbus_dispatcherWakeup(conn->dispatcher);
    }
}


cdbus_Dispatcher*
cdbus_dispatcherNew
    (
    cdbus_MainLoop*     loop
    )
{
    cdbus_Dispatcher* dispatcher = NULL;
    cdbus_Bool status;
    cdbus_Descriptor pipeReadFd = -1;

    cdbus_ptrPtrMapLock(cdbus_gDispatcherRegistry);
    /* See if there is another dispatcher that is already registered
     * with the same event loop.
     */
    dispatcher = cdbus_ptrPtrMapGet(cdbus_gDispatcherRegistry, loop);
    cdbus_ptrPtrMapUnlock(cdbus_gDispatcherRegistry);

    if ( NULL != dispatcher )
    {
        /* A matching dispatcher already exists so we'll add a reference
         * and return that as the dispatcher.
         */
        cdbus_dispatcherRef(dispatcher);
    }
    /* Else if this is a valid loop then ... */
    else if ( NULL != loop )
    {
        /* Create a new dispatcher for this loop */
        dispatcher = cdbus_calloc(1, sizeof(*dispatcher));
        if ( NULL != dispatcher )
        {
            dispatcher->loop = loop->loopRef(loop);
            dispatcher->exitLoop = CDBUS_FALSE;

            dispatcher->finalizerFunc = NULL;
            dispatcher->finalizerData = NULL;

            LIST_INIT(&dispatcher->connections);
            LIST_INIT(&dispatcher->watches);
            LIST_INIT(&dispatcher->timeouts);
            CDBUS_LOCK_ALLOC(dispatcher->lock, CDBUS_MUTEX_RECURSIVE);

            if ( CDBUS_LOCK_IS_NULL(dispatcher->lock) )
            {
                dispatcher->loop->loopUnref(dispatcher->loop);
                cdbus_free(dispatcher);
                dispatcher = NULL;
            }


            if ( NULL != dispatcher )
            {
                dispatcher->wakeupPipe = cdbus_pipeNew();
                if ( NULL != dispatcher->wakeupPipe )
                {
                    cdbus_pipeGetFds(dispatcher->wakeupPipe, &pipeReadFd, NULL);
                    dispatcher->wakeupWatch = loop->watchNew(loop,
                                                        pipeReadFd,
                                                        DBUS_WATCH_READABLE,
                                                        cdbus_wakeupHandler,
                                                        dispatcher);
                    loop->watchStart(dispatcher->wakeupWatch);
                    dispatcher->wakeupTrigger = CDBUS_FALSE;
                    dispatcher->dispatchNeeded = CDBUS_FALSE;
                }

                /* If something went wrong allocating these things then ... */
                if ( (NULL == dispatcher->wakeupPipe) ||
                    (NULL == dispatcher->wakeupWatch) )
                {
                    if ( NULL != dispatcher->wakeupWatch )
                    {
                        loop->watchDestroy(dispatcher->wakeupWatch);
                    }

                    if ( NULL != dispatcher->wakeupPipe )
                    {
                        cdbus_pipeDestroy(dispatcher->wakeupPipe);
                    }

                    if ( !CDBUS_LOCK_IS_NULL(dispatcher->lock) )
                    {
                        CDBUS_LOCK_FREE(dispatcher->lock);
                    }

                    dispatcher->loop->loopUnref(dispatcher->loop);
                    cdbus_free(dispatcher);
                    dispatcher = NULL;
                }
            }


            if ( NULL != dispatcher )
            {
                /* Add the first reference to the dispatcher */
                cdbus_dispatcherRef(dispatcher);

                CDBUS_TRACE((CDBUS_TRC_INFO,
                    "Created dispatcher instance (%p)", (void*)dispatcher));

                /* Register the loop to the dispatcher */
                cdbus_ptrPtrMapLock(cdbus_gDispatcherRegistry);
                status = cdbus_ptrPtrMapAdd(cdbus_gDispatcherRegistry,
                                            loop, dispatcher);
                cdbus_ptrPtrMapUnlock(cdbus_gDispatcherRegistry);

                if ( !status )
                {
                    CDBUS_TRACE((CDBUS_TRC_ERROR,
                                "Failed to add dispatcher to registry!"));
                }
            }
        }
    }

    return dispatcher;
}


void
cdbus_dispatcherUnref
    (
    cdbus_Dispatcher*   dispatcher
    )
{
    cdbus_Int32 value = 0;
    cdbus_Connection* conn = NULL;
    cdbus_Connection* nextConn = NULL;
    void* mapValue = NULL;

    assert( NULL != dispatcher );
    if ( NULL != dispatcher )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&dispatcher->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            CDBUS_LOCK(dispatcher->lock);

            /* Destroy all the connections */
            for ( conn = LIST_FIRST(&dispatcher->connections);
                conn != LIST_END(&dispatcher->connections);
                conn = nextConn )
            {
                nextConn = LIST_NEXT(conn, link);
                cdbus_dispatcherRemoveConnection(dispatcher, conn);
            }

            /* Remove the loop/dispatcher mapping from the global registry */
            cdbus_ptrPtrMapLock(cdbus_gDispatcherRegistry);
            mapValue = cdbus_ptrPtrMapRemove(cdbus_gDispatcherRegistry,
                                            dispatcher->loop);

#ifdef NDEBUG
            CDBUS_UNUSED(mapValue);
#endif
            assert( mapValue == (void*)dispatcher);

            cdbus_ptrPtrMapUnlock(cdbus_gDispatcherRegistry);

            /* Destroy the dispatcher itself */
            if ( NULL != dispatcher->loop )
            {
                dispatcher->loop->watchStop(dispatcher->wakeupWatch);
                dispatcher->loop->watchDestroy(dispatcher->wakeupWatch);
                cdbus_pipeDestroy(dispatcher->wakeupPipe);

                dispatcher->loop->loopUnref(dispatcher->loop);
                CDBUS_TRACE((CDBUS_TRC_INFO,
                            "Unreferenced dispatcher loop"));
            }

            /* If there is a hooked defined that needs to know when the
             * Dispatcher is truly destroyed then call it now!
             */
            if ( NULL != dispatcher->finalizerFunc )
            {
                dispatcher->finalizerFunc(dispatcher->finalizerData);
            }

            CDBUS_UNLOCK(dispatcher->lock);

            CDBUS_LOCK_FREE(CDBUS_DISPATCHER_A->lock);

            /* Free the dispatcher itself */
            cdbus_free(dispatcher);
            CDBUS_TRACE((CDBUS_TRC_INFO,
                 "Destroyed dispatcher instance (%p)", (void*)dispatcher));
        }
    }
}


cdbus_Dispatcher*
cdbus_dispatcherRef
    (
    cdbus_Dispatcher* dispatcher
    )
{
    if ( NULL != dispatcher )
    {
        cdbus_atomicAdd(&dispatcher->refCnt, 1);
    }

    return dispatcher;
}


cdbus_HResult
cdbus_dispatcherAddConnection
    (
    cdbus_Dispatcher*           dispatcher,
    struct cdbus_Connection*    conn
    )
{
    cdbus_Connection*   curConn = NULL;
    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                            CDBUS_FAC_CDBUS,
                            CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != conn) && (NULL != dispatcher) )
    {
        CDBUS_LOCK(dispatcher->lock);

        /* Only add the connection if it doesn't already exist in the list */
        LIST_FOREACH(curConn, &dispatcher->connections, link)
        {
            if ( curConn == conn )
            {
                break;
            }
        }

        /* If the connection is already in the list of connections then ... */
        if ( curConn != LIST_END(&dispatcher->connections) )
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
                LIST_INSERT_HEAD(&dispatcher->connections,
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
                    dbus_connection_get_dispatch_status(
                    cdbus_connectionGetDBus(conn)) )
                {
                    dispatcher->dispatchNeeded = CDBUS_TRUE;
                }
                cdbus_dispatcherWakeup(dispatcher);
            }
        }
        CDBUS_UNLOCK(dispatcher->lock);
    }

    return status;
}


cdbus_HResult
cdbus_dispatcherRemoveConnection
    (
    cdbus_Dispatcher*           dispatcher,
    struct cdbus_Connection*    conn
    )
{
    cdbus_Connection* curConn = NULL;

    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != conn) && (NULL != dispatcher) )
    {
        CDBUS_LOCK(dispatcher->lock);
        LIST_FOREACH(curConn, &dispatcher->connections, link)
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
        if ( LIST_END(&dispatcher->connections) == curConn )
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
            cdbus_dispatcherWakeup(dispatcher);
        }
        CDBUS_UNLOCK(dispatcher->lock);
    }

    return status;
}


struct cdbus_Connection*
cdbus_dispatcherGetDbusConnOwner
    (
    cdbus_Dispatcher*   dispatcher,
    DBusConnection*     dbusConn
    )
{
    cdbus_Connection* curConn = NULL;

    if ( NULL != dbusConn )
    {
        CDBUS_LOCK(dispatcher->lock);
        LIST_FOREACH(curConn, &dispatcher->connections, link)
        {
            if ( curConn->dbusConn == dbusConn )
            {
                break;
            }
        }

        /* If a match is not found then curConn is NULL at
         * the end of this loop
         */
        CDBUS_UNLOCK(dispatcher->lock);
    }

    return curConn;
}


cdbus_HResult
cdbus_dispatcherAddWatch
    (
    cdbus_Dispatcher*   dispatcher,
    struct cdbus_Watch* watch
    )
{
    cdbus_Watch* curWatch = NULL;
    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                            CDBUS_FAC_CDBUS,
                            CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != watch) && (NULL != dispatcher) )
    {
        CDBUS_LOCK(dispatcher->lock);
        /* Only add the watch if it doesn't already exist in the list */
        LIST_FOREACH(curWatch, &dispatcher->watches, link)
        {
            if ( curWatch == watch )
            {
                break;
            }
        }

        /* If the connection isn't already in the list of connections
         * then ...
         */
        if ( curWatch == LIST_END(&dispatcher->watches) )
        {
            LIST_INSERT_HEAD(&dispatcher->watches, watch, link);

            /* This dispatcher now references it too */
            cdbus_watchRef(watch);
        }
        CDBUS_UNLOCK(dispatcher->lock);

        status = CDBUS_RESULT_SUCCESS;
    }

    return status;
}


cdbus_HResult
cdbus_dispatcherRemoveWatch
    (
    cdbus_Dispatcher*   dispatcher,
    struct cdbus_Watch* watch
    )
{
    cdbus_Watch* curWatch = NULL;

    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != watch) && (NULL != dispatcher) )
    {
        CDBUS_LOCK(dispatcher->lock);
        LIST_FOREACH(curWatch, &dispatcher->watches, link)
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
        if ( LIST_END(&dispatcher->watches) == curWatch )
        {
            status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_NOT_FOUND);
        }
        CDBUS_UNLOCK(dispatcher->lock);
    }

    return status;
}


cdbus_HResult
cdbus_dispatcherAddTimeout
    (
    cdbus_Dispatcher*       dispatcher,
    struct cdbus_Timeout*   timeout
    )
{
    cdbus_Timeout* curTimeout = NULL;
    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                            CDBUS_FAC_CDBUS,
                            CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != timeout) && (NULL != dispatcher) )
    {
        CDBUS_LOCK(dispatcher->lock);
        /* Only add the watch if it doesn't already exist in the list */
        LIST_FOREACH(curTimeout, &dispatcher->timeouts, link)
        {
            if ( curTimeout == timeout )
            {
                break;
            }
        }

        /* If the connection isn't already in the list of connections
         * then ...
         */
        if ( curTimeout == LIST_END(&dispatcher->timeouts) )
        {
            LIST_INSERT_HEAD(&dispatcher->timeouts, timeout, link);
            /* The dispatcher now holds a reference to the timeout */
            cdbus_timeoutRef(timeout);
        }
        CDBUS_UNLOCK(dispatcher->lock);

        status = CDBUS_RESULT_SUCCESS;
    }

    return status;
}


cdbus_HResult
cdbus_dispatcherRemoveTimeout
    (
    cdbus_Dispatcher*       dispatcher,
    struct cdbus_Timeout*   timeout
    )
{
    cdbus_Timeout* curTimeout = NULL;

    cdbus_HResult status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_INVALID_PARAMETER);

    if ( (NULL != timeout) && (NULL != dispatcher) )
    {
        CDBUS_LOCK(dispatcher->lock);
        LIST_FOREACH(curTimeout, &dispatcher->timeouts, link)
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
        if ( LIST_END(&dispatcher->timeouts) == curTimeout )
        {
            status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                        CDBUS_FAC_CDBUS,
                                        CDBUS_EC_NOT_FOUND);
        }
        CDBUS_UNLOCK(dispatcher->lock);
    }

    return status;
}


void cdbus_dispatcherTriggerWakeup
    (
    cdbus_Dispatcher* dispatcher
    )
{
    const cdbus_Char buf = 'W';

    if ( NULL != dispatcher )
    {
        /* If the wakeup function hasn't been triggered yet then */
        if ( !dispatcher->wakeupTrigger )
        {
            if ( sizeof(buf) == cdbus_pipeWrite(dispatcher->wakeupPipe, &buf,
                sizeof(buf)) )
            {
                dispatcher->wakeupTrigger = CDBUS_TRUE;
            }
            else
            {
                CDBUS_TRACE((CDBUS_TRC_WARN,
                           "Failed to write wakeup trigger to pipe"));
            }
        }
    }
}


void cdbus_dispatcherWakeup
    (
    cdbus_Dispatcher* dispatcher
    )
{
    if ( NULL != dispatcher )
    {
        CDBUS_LOCK(dispatcher->lock);
        cdbus_dispatcherTriggerWakeup(dispatcher);
        CDBUS_UNLOCK(dispatcher->lock);
    }
}


cdbus_HResult
cdbus_dispatcherRun
    (
    cdbus_Dispatcher*   dispatcher,
    cdbus_RunOption     runOpt
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    cdbus_Bool canBlock = CDBUS_TRUE;

    if ( NULL == dispatcher )
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
                canBlock = CDBUS_TRUE;
                break;

            case CDBUS_RUN_NO_WAIT:
                canBlock = CDBUS_FALSE;
                break;

            case CDBUS_RUN_ONCE:
                canBlock = CDBUS_TRUE;
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
            dispatcher->exitLoop = CDBUS_FALSE;
            do
            {
                CDBUS_LOCK(dispatcher->lock);
                dispatcher->loop->loopIterate(dispatcher->loop, canBlock);
                CDBUS_UNLOCK(dispatcher->lock);
            }
            while ( !dispatcher->exitLoop &&
                (CDBUS_RUN_WAIT == runOpt) );

            /* Keeps the state consistent now that the loop
             * has exited.
             */
            dispatcher->exitLoop = CDBUS_TRUE;
        }
    }

    return rc;
}


cdbus_HResult
cdbus_dispatcherStop
    (
    cdbus_Dispatcher*   dispatcher
    )
{
    /* This function purposely does not try to lock
     * the dispatcher so that it is safe to be called
     * from a signal handler. This may make it a little
     * thread-unsafe so it shouldn't be called when another
     * thread might be trying to destroy the dispatcher or
     * start it running again.
     */
    dispatcher->exitLoop = CDBUS_TRUE;
    /* Force the handler to scan through it's connections
     * and flush everything.
     */
    dispatcher->dispatchNeeded = CDBUS_TRUE;
    cdbus_dispatcherTriggerWakeup(dispatcher);

    return CDBUS_RESULT_SUCCESS;
}


void
cdbus_dispatcherBreak
    (
    cdbus_Dispatcher*   dispatcher
    )
{
    cdbus_dispatcherStop(dispatcher);
    dispatcher->loop->loopQuit(dispatcher->loop);
}


void
cdbus_dispatcherSetFinalizer
    (
    cdbus_Dispatcher*   dispatcher,
    cdbus_FinalizerFunc finalizer,
    void*               data
    )
{
    if ( NULL != dispatcher )
    {
        dispatcher->finalizerFunc = finalizer;
        dispatcher->finalizerData = data;
    }
}
