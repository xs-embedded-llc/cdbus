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
 * @file           timeout-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of the timeout class.
 *******************************************************************************
 */

#ifndef CDBUS_TIMEOUT_PRIV_C_
#define CDBUS_TIMEOUT_PRIV_C_

#include <stddef.h>
#include <assert.h>
#include "cdbus/timeout.h"
#include "cdbus/dispatcher.h"
#include "timeout-priv.h"
#include "dispatcher-priv.h"
#include "atomic-ops.h"
#include "alloc.h"
#include "trace.h"

static void
cdbus_timerCallback
    (
    EV_P_
    ev_timer*   evTimer,
    int         rcvEvents
    )
{
    cdbus_Timeout* timeout = evTimer->data;
    cdbus_TimeoutHandler handler = NULL;
    void* data = NULL;

    if ( NULL == timeout )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Can't cast to timeout in timer event callback"));
    }
    else
    {
        /* Make a copy while we hold the lock so we don't have to
         * maintain the lock while calling the handler.
         */
        CDBUS_LOCK(timeout->lock);
        handler = timeout->handler;
        data = timeout->data;
        CDBUS_UNLOCK(timeout->lock);

        if ( NULL != handler )
        {
            handler(timeout, data);
        }
        else
        {
            CDBUS_TRACE((CDBUS_TRC_INFO, "No timeout handler configured"));
        }

        /* Enable or disable the timer based on the repeat mode */
        cdbus_timeoutEnable(timeout, cdbus_timeoutGetRepeat(timeout));
    }
}


cdbus_Timeout*
cdbus_timeoutNew
    (
    struct cdbus_Dispatcher*    dispatcher,
    cdbus_Int32                 msecInterval,
    cdbus_Bool                  repeat,
    cdbus_TimeoutHandler        h,
    void*                       data
    )
{
    cdbus_Timeout* timeout = NULL;
    ev_tstamp period = 0;

    if ( NULL != dispatcher )
    {
        timeout = cdbus_calloc(1, sizeof(*timeout));
        if ( NULL != timeout )
        {
            timeout->lock = cdbus_mutexNew(CDBUS_MUTEX_RECURSIVE);
            if ( NULL == timeout->lock )
            {
                cdbus_free(timeout);
                timeout = NULL;
            }
            else
            {
                period = (ev_tstamp)msecInterval / 1000.0;
                ev_timer_init(&timeout->timerWatcher, cdbus_timerCallback,
                              0.0, period);
                timeout->dispatcher = cdbus_dispatcherRef(dispatcher);
                timeout->data = data;
                timeout->timerWatcher.data = timeout;
                timeout->handler = h;
                timeout->repeat = repeat;
                timeout = cdbus_timeoutRef(timeout);
                CDBUS_TRACE((CDBUS_TRC_INFO,
                      "Created timeout instance (%p)", (void*)timeout));
            }
        }
    }

    return timeout;
}


cdbus_Timeout*
cdbus_timeoutRef
    (
    cdbus_Timeout*  t
    )
{
    assert( t != NULL );
    if ( NULL != t )
    {
        cdbus_atomicAdd(&t->refCnt, 1);
    }

    return t;
}


void
cdbus_timeoutUnref
    (
    cdbus_Timeout*  t
    )
{
    cdbus_Int32 value;
    assert( NULL != t );
    if ( NULL != t )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&t->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            CDBUS_LOCK(t->lock);
            if ( ev_is_active(&t->timerWatcher) )
            {
#if EV_MULTIPLICITY
                ev_timer_stop(t->dispatcher->EV_A, &t->timerWatcher);
#else
                ev_timer_stop(&t->timerWatcher);
#endif
            }
            cdbus_dispatcherUnref(t->dispatcher);
            CDBUS_UNLOCK(t->lock);
            cdbus_mutexFree(t->lock);
            cdbus_free(t);
            CDBUS_TRACE((CDBUS_TRC_INFO,
                  "Destroyed timeout instance (%p)", (void*)t));
        }
    }
}


cdbus_Bool
cdbus_timeoutIsEnabled
    (
    cdbus_Timeout*  t
    )
{
    cdbus_UInt32 isEnabled = CDBUS_FALSE;

    assert( t != NULL );
    if ( NULL == t )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (t == NULL)"));
    }
    else
    {
        CDBUS_LOCK(t->lock);
        isEnabled = ev_is_active(&t->timerWatcher);
        CDBUS_UNLOCK(t->lock);
    }

    return isEnabled;
}


cdbus_HResult
cdbus_timeoutEnable
    (
    cdbus_Timeout*  t,
    cdbus_Bool      option
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;

    assert( t != NULL );
    if ( NULL == t )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (t == NULL)"));
        rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                CDBUS_FAC_CDBUS,
                                CDBUS_EC_INVALID_PARAMETER);
    }
    else
    {
        CDBUS_LOCK(t->lock);

        /* If we want to enable the watcher then ... */
        if ( option )
        {
            /* If it's not currently active (and thus managed by
             * the dispatcher) then ...
             */
            if ( !ev_is_active(&t->timerWatcher) )
            {
                /* This has the side-effect of activating the timer */
                rc = cdbus_dispatcherAddTimeout(t->dispatcher, t);
                if ( !CDBUS_SUCCEEDED(rc) )
                {
                    CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to add timeout!"));
                }
            }
        }
        else
        {
            /* If the watch is currently active then
             * this implies that the dispatcher is managing it.
             */
            if ( ev_is_active(&t->timerWatcher) )
            {
                /* This will disable the watcher */
                rc = cdbus_dispatcherRemoveTimeout(t->dispatcher, t);
                if ( !CDBUS_SUCCEEDED(rc) )
                {
                    CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to remove timeout!"));
                }
            }
        }
        CDBUS_UNLOCK(w->lock);
    }

    return rc;
}


cdbus_Int32
cdbus_timeoutInterval
    (
    cdbus_Timeout*  t
    )
{
    cdbus_UInt32 period = 0;

    assert( t != NULL );

    if ( NULL == t )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (t == NULL)"));
    }
    else
    {
        CDBUS_LOCK(t->lock);
        period = (cdbus_Int32)(t->timerWatcher.repeat * 1000);
        CDBUS_UNLOCK(t->lock);
    }

    return period;
}


cdbus_HResult
cdbus_timeoutSetInterval
    (
    cdbus_Timeout*  t,
    cdbus_Int32     msecInterval
    )
{
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    ev_tstamp period = 0.0;

    assert( t != NULL );

    if ( NULL == t )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (t == NULL)"));
        rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                CDBUS_FAC_CDBUS,
                                CDBUS_EC_INVALID_PARAMETER);
    }
    else
    {
        CDBUS_LOCK(t->lock);
        period = (ev_tstamp)msecInterval / 1000.0;
        /* If the watch is active (and being managed by the dispatcher) then ... */
        if ( ev_is_active(&t->timerWatcher) )
        {
            /* You can't change the internal of an active watcher */
            rc = cdbus_dispatcherRemoveTimeout(t->dispatcher, t);
            if ( !CDBUS_SUCCEEDED(rc) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to remove timeout!"));
            }
            else
            {
                /* Change the flags and now re-install the watcher to enable it again */
                ev_timer_set(&t->timerWatcher, 0.0, period);
                rc = cdbus_dispatcherAddTimeout(t->dispatcher, t);
                if ( !CDBUS_SUCCEEDED(rc) )
                {
                    CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to add timeout!"));
                }
            }
        }
        /* Else the watcher is not active and being managed */
        else
        {
            /* Directly update the internal since it's not running */
            ev_timer_set(&t->timerWatcher, 0.0, period);
        }
        CDBUS_UNLOCK(w->lock);
    }

    return rc;
}


void*
cdbus_timeoutGetData
    (
    cdbus_Timeout*  t
    )
{
    void* data = NULL;

    assert( t != NULL );
    if ( NULL == t )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (t == NULL)"));
    }
    else
    {
        CDBUS_LOCK(t->lock);
        data = t->data;
        CDBUS_UNLOCK(t->lock);
    }

    return data;
}


void
cdbus_timeoutSetData
    (
    cdbus_Timeout*  t,
    void*           data
    )
{
    assert( t != NULL );
    if ( NULL == t )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (t == NULL)"));
    }
    else
    {
        CDBUS_LOCK(t->lock);
        t->data = data;
        CDBUS_UNLOCK(t->lock);
    }
}


cdbus_Bool
cdbus_timeoutGetRepeat
    (
    cdbus_Timeout*  t
    )
{
    cdbus_Bool repeat = CDBUS_FALSE;
    assert( t != NULL );
    if ( NULL == t )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (t == NULL)"));
    }
    else
    {
        CDBUS_LOCK(t->lock);
        repeat = t->repeat;
        CDBUS_UNLOCK(t->lock);
    }

    return repeat;
}


void
cdbus_timeoutSetRepeat
    (
    cdbus_Timeout*  t,
    cdbus_Bool      repeat
    )
{
    assert( t != NULL );
    if ( NULL == t )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR, "Invalid parameter (t == NULL)"));
    }
    else
    {
        CDBUS_LOCK(t->lock);
        t->repeat = repeat;
        CDBUS_UNLOCK(t->lock);
    }
}

#endif /* Guard for CDBUS_TIMEOUT_PRIV_C_ */
