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
 * @file           watch-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of the watch class.
 *===========================================================================
 */
#include <stddef.h>
#include <assert.h>
#include "cdbus/watch.h"
#include "cdbus/dispatcher.h"
#include "cdbus/atomic-ops.h"
#include "cdbus/alloc.h"
#include "cdbus/mainloop.h"
#include "watch-priv.h"
#include "dispatcher-priv.h"
#include "trace.h"
#include "internal.h"


static void
cdbus_watchCallback
    (
    cdbus_MainLoopWatch*    watch,
    cdbus_UInt32            flags,
    void*                   userData
    )
{
    cdbus_Watch* w = (cdbus_Watch*)userData;
    cdbus_WatchHandler handler = NULL;
    void* data = NULL;

    CDBUS_UNUSED(watch);

    if ( NULL == w )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR,
            "Can't cast to watch in event callback"));
    }
    else
    {
        /* Add a reference to the watch in case
         * the callback handler tries to unreference
         * it.
         */
        cdbus_watchRef(w);

        /* Make a copy while we hold the lock so we don't have to
         * maintain the lock while calling the handler.
         */
        CDBUS_LOCK(w->lock);
        handler = w->handler;
        data = w->data;
        CDBUS_UNLOCK(w->lock);

        if ( NULL != handler )
        {
            handler(w, flags, data);
        }
        else
        {
            CDBUS_TRACE((CDBUS_TRC_INFO, "No watch handler configured"));
        }

        cdbus_watchUnref(w);
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
    cdbus_Watch* w = NULL;

    if ( NULL != dispatcher )
    {
        w = cdbus_calloc(1, sizeof(*w));
        if ( NULL != w )
        {
            CDBUS_LOCK_ALLOC(w->lock, CDBUS_MUTEX_RECURSIVE);
            if ( CDBUS_LOCK_IS_NULL(w->lock) )
            {
                cdbus_free(w);
                w = NULL;
            }
            else
            {
                w->watch = dispatcher->loop->watchNew(
                                        dispatcher->loop,
                                        fd,
                                        flags,
                                        cdbus_watchCallback,
                                        w);
                if ( NULL == w->watch )
                {
                    CDBUS_LOCK_FREE(w->lock);
                    cdbus_free(w);
                    w = NULL;
                }
                else
                {
                    w->dispatcher = cdbus_dispatcherRef(dispatcher);
                    w->data = data;
                    w->handler = h;
                    w = cdbus_watchRef(w);
                    CDBUS_TRACE((CDBUS_TRC_INFO,
                         "Created watch instance (%p)", (void*)w));
                }
            }
        }
    }

    return w;
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
            if ( w->dispatcher->loop->watchIsEnabled(w->watch) )
            {
                w->dispatcher->loop->watchStop(w->watch);
            }
            w->dispatcher->loop->watchDestroy(w->watch);
            cdbus_dispatcherUnref(w->dispatcher);
            CDBUS_UNLOCK(w->lock);
            CDBUS_LOCK_FREE(w->lock);
            cdbus_free(w);
            CDBUS_TRACE((CDBUS_TRC_INFO,
                  "Destroyed watch instance (%p)", (void*)w));
        }
    }
}


cdbus_Descriptor
cdbus_watchGetDescriptor
    (
    cdbus_Watch*    w
    )
{
    cdbus_Descriptor fd = -1;

    if ( NULL != w )
    {
        fd = w->dispatcher->loop->watchGetDescriptor(w->watch);
    }

    return fd;
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
        flags = w->dispatcher->loop->watchGetFlags(w->watch);
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
        /* If the watch is active then ... */
        if ( w->dispatcher->loop->watchIsEnabled(w->watch) )
        {
            /* You can't change the flags of an active watcher */
            w->dispatcher->loop->watchStop(w->watch);

            /* Change the flags */
            w->dispatcher->loop->watchSetFlags(w->watch, flags);

            /* Now start the watcher again */
            w->dispatcher->loop->watchStart(w->watch);
        }
        /* Else the watcher is not active and being managed */
        else
        {
            /* Directly update the flags since it's not running */
            w->dispatcher->loop->watchSetFlags(w->watch, flags);
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
        isEnabled = w->dispatcher->loop->watchIsEnabled(w->watch);
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
            /* If it's not currently active then ...
             */
            if ( !w->dispatcher->loop->watchIsEnabled(w->watch) )
            {
                /* Start the watch */
                w->dispatcher->loop->watchStart(w->watch);
                cdbus_dispatcherWakeup(w->dispatcher);
            }
        }
        else
        {
            /* If the watch is currently active then ...
             */
            if ( w->dispatcher->loop->watchIsEnabled(w->watch) )
            {
                /* Stop the watch */
                w->dispatcher->loop->watchStop(w->watch);
                cdbus_dispatcherWakeup(w->dispatcher);
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
