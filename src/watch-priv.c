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
 * @file           watch-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of the watch class.
 *******************************************************************************
 */
#include <stddef.h>
#include <assert.h>
#include "cdbus/watch.h"
#include "cdbus/dispatcher.h"
#include "atomic-ops.h"
#include "watch-priv.h"
#include "dispatcher-priv.h"
#include "alloc.h"
#include "trace.h"

static cdbus_UInt32
cdbus_convertToEvFlags
    (
    cdbus_UInt32 dbusFlags
    )
{
    cdbus_UInt32 evFlags = 0U;

    if ( dbusFlags & DBUS_WATCH_READABLE ) evFlags |= EV_READ;
    if ( dbusFlags & DBUS_WATCH_WRITABLE ) evFlags |= EV_WRITE;

    if ( dbusFlags & DBUS_WATCH_ERROR )
    {
        evFlags |= (EV_READ | EV_WRITE);
    }

    if ( dbusFlags & DBUS_WATCH_HANGUP )
    {
        evFlags |= (EV_READ | EV_WRITE);
    }

    return evFlags;
}


static cdbus_UInt32
cdbus_convertToDbusFlags
    (
    cdbus_UInt32 evFlags
    )
{
    cdbus_UInt32 dbusFlags = 0U;

    if ( evFlags & EV_READ ) dbusFlags |= DBUS_WATCH_READABLE;
    if ( evFlags & EV_WRITE ) dbusFlags |= DBUS_WATCH_WRITABLE;

    return dbusFlags;
}


static void
cdbus_ioWatchCallback
    (
    EV_P_
    ev_io*  evIo,
    int rcvEvents
    )
{
    cdbus_Watch* w = evIo->data;
    cdbus_WatchHandler handler = NULL;
    void* data = NULL;
    if ( NULL == w )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Can't cast to watch in IO event callback"));
    }
    else
    {
        /* Make a copy while we hold the lock so we don't have to
         * maintain the lock while calling the handler.
         */
        CDBUS_LOCK(w->lock);
        handler = w->handler;
        data = w->data;
        CDBUS_UNLOCK(w->lock);

        if ( NULL != handler )
        {
            handler(w, cdbus_convertToDbusFlags(rcvEvents), data);
        }
        else
        {
            CDBUS_TRACE((CDBUS_TRC_INFO, "No watch handler configured"));
        }
    }
}


cdbus_Watch*
cdbus_watchNew
    (
    struct cdbus_Dispatcher*    dispatcher,
    cdbus_Descriptor            fd,
    cdbus_UInt32                flags,
    cdbus_WatchHandler          h,
    void*                       data
    )
{
    cdbus_Watch* watch = NULL;

    if ( NULL != dispatcher )
    {
        watch = cdbus_calloc(1, sizeof(*watch));
        if ( NULL != watch )
        {
            watch->lock = cdbus_mutexNew(CDBUS_MUTEX_RECURSIVE);
            if ( NULL == watch->lock )
            {
                cdbus_free(watch);
                watch = NULL;
            }
            else
            {
                ev_io_init(&watch->ioWatcher, cdbus_ioWatchCallback,
                            fd, cdbus_convertToEvFlags(flags));
                watch->dispatcher = cdbus_dispatcherRef(dispatcher);
                watch->data = data;
                watch->ioWatcher.data = watch;
                watch->handler = h;
                watch = cdbus_watchRef(watch);
                CDBUS_TRACE((CDBUS_TRC_INFO,
                     "Created watch instance (%p)", (void*)watch));
            }
        }
    }

    return watch;
}


cdbus_Watch*
cdbus_watchRef
    (
    cdbus_Watch*    w
    )
{
    assert( w != NULL );

    if ( NULL != w )
    {
        cdbus_atomicAdd(&w->refCnt, 1);
    }

    return w;
}


void
cdbus_watchUnref
    (
    cdbus_Watch*    w
    )
{
    cdbus_Int32 value;
    assert( NULL != w );
    if ( NULL != w )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&w->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            CDBUS_LOCK(w->lock);
            if ( ev_is_active(&w->ioWatcher) )
            {
#if EV_MULTIPLICITY
                ev_io_stop(w->dispatcher->EV_A, &w->ioWatcher);
#else
                ev_io_stop(&w->ioWatcher);
#endif
            }
            cdbus_dispatcherUnref(w->dispatcher);
            CDBUS_UNLOCK(w->lock);
            cdbus_mutexFree(w->lock);
            cdbus_free(w);
            CDBUS_TRACE((CDBUS_TRC_INFO,
                  "Destroyed watch instance (%p)", (void*)w));
        }
    }
}


cdbus_UInt32
cdbus_watchGetFlags
    (
    cdbus_Watch*    w
    )
{
    cdbus_UInt32 flags = 0;

    assert( w != NULL );

    if ( NULL == w )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (w == NULL)"));
    }
    else
    {
        CDBUS_LOCK(w->lock);
        flags = cdbus_convertToDbusFlags(w->ioWatcher.events);
        CDBUS_UNLOCK(w->lock);
    }

    return flags;
}


cdbus_HResult
cdbus_watchSetFlags
    (
    cdbus_Watch*    w,
    cdbus_UInt32    flags
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;

    assert( w != NULL );

    if ( NULL == w )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (w == NULL)"));
        rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                CDBUS_FAC_CDBUS,
                                CDBUS_EC_INVALID_PARAMETER);
    }
    else
    {
        CDBUS_LOCK(w->lock);
        /* If the watch is active (and being managed by the dispatcher) then ... */
        if ( ev_is_active(&w->ioWatcher) )
        {
            /* You can't change the flags of an active watcher */
            rc = cdbus_dispatcherRemoveWatch(w->dispatcher, w);
            if ( !CDBUS_SUCCEEDED(rc) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to remove watch!"));
            }
            else
            {
                /* Change the flags and now re-install the watcher to enable it again */
                ev_io_set(&w->ioWatcher, w->ioWatcher.fd, cdbus_convertToEvFlags(flags));
                rc = cdbus_dispatcherAddWatch(w->dispatcher, w);
                if ( !CDBUS_SUCCEEDED(rc) )
                {
                    CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to add watch!"));
                }
            }
        }
        /* Else the watcher is not active and being managed */
        else
        {
            /* Directly update the flags since it's not running */
            ev_io_set(&w->ioWatcher, w->ioWatcher.fd, cdbus_convertToEvFlags(flags));
        }
        CDBUS_UNLOCK(w->lock);
    }

    return rc;
}


cdbus_Bool
cdbus_watchIsEnabled
    (
    cdbus_Watch*    w
    )
{
    cdbus_UInt32 isEnabled = CDBUS_FALSE;

    assert( w != NULL );

    if ( NULL == w )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (w == NULL)"));
    }
    else
    {
        CDBUS_LOCK(w->lock);
        isEnabled = ev_is_active(&w->ioWatcher);
        CDBUS_UNLOCK(w->lock);
    }

    return isEnabled;
}


cdbus_HResult
cdbus_watchEnable
    (
    cdbus_Watch*    w,
    cdbus_Bool      option
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;

    assert( w != NULL );

    if ( NULL == w )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (w == NULL)"));
        rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                CDBUS_FAC_CDBUS,
                                CDBUS_EC_INVALID_PARAMETER);
    }
    else
    {
        CDBUS_LOCK(w->lock);

        /* If we want to enable the watcher then ... */
        if ( option )
        {
            /* If it's not currently active (and thus managed by
             * the dispatcher) then ...
             */
            if ( !ev_is_active(&w->ioWatcher) )
            {
                /* This has the side-effect of activating the watch */
                rc = cdbus_dispatcherAddWatch(w->dispatcher, w);
                if ( !CDBUS_SUCCEEDED(rc) )
                {
                    CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to add watch!"));
                }
            }
        }
        else
        {
            /* If the watch is currently active then
             * this implies that the dispatcher is managing it.
             */
            if ( ev_is_active(&w->ioWatcher) )
            {
                /* This will disable the watcher */
                rc = cdbus_dispatcherRemoveWatch(w->dispatcher, w);
                if ( !CDBUS_SUCCEEDED(rc) )
                {
                    CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to remove watch!"));
                }
            }
        }
        CDBUS_UNLOCK(w->lock);
    }

    return rc;
}


void*
cdbus_watchGetData
    (
    cdbus_Watch*    w
    )
{
    assert( w != NULL );

    void* data = NULL;

    if ( NULL == w )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (w == NULL)"));
    }
    else
    {
        CDBUS_LOCK(w->lock);
        data = w->data;
        CDBUS_UNLOCK(w->lock);
    }

    return data;
}


void
cdbus_watchSetData
    (
    cdbus_Watch*    w,
    void*           data
    )
{
    assert( w != NULL );

    if ( NULL == w )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (w == NULL)"));
    }
    else
    {
        CDBUS_LOCK(w->lock);
        w->data = data;
        CDBUS_UNLOCK(w->lock);
    }
}


