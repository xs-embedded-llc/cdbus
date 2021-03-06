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
 * @file           timeout-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of the timeout class.
 *===========================================================================
 */

#ifndef CDBUS_TIMEOUT_PRIV_C_
#define CDBUS_TIMEOUT_PRIV_C_

#include <stddef.h>
#include <assert.h>
#include "cdbus/timeout.h"
#include "cdbus/dispatcher.h"
#include "cdbus/atomic-ops.h"
#include "cdbus/alloc.h"
#include "cdbus/mainloop.h"
#include "timeout-priv.h"
#include "dispatcher-priv.h"
#include "trace.h"
#include "internal.h"

static void
cdbus_timerCallback
    (
    cdbus_MainLoopTimer*    timer,
    void*                   userData
    )
{
    cdbus_Timeout* t = (cdbus_Timeout*)userData;
    cdbus_TimeoutHandler handler = NULL;
    cdbus_MainLoop* loop;
    void* data = NULL;

    if ( NULL == t )
    {
        CDBUS_TRACE((CDBUS_TRC_ERROR,
            "Can't cast to timeout in timer event callback"));
    }
    else
    {
        /* Add a reference to the timer in case
         * the callback handler tries to unreference
         * it.
         */
        cdbus_timeoutRef(t);

        /* Make a copy while we hold the lock so we don't have to
         * maintain the lock while calling the handler.
         */
        CDBUS_LOCK(t->lock);
        handler = t->handler;
        data = t->data;
        loop = t->dispatcher->loop;

        if ( NULL != handler )
        {
            CDBUS_UNLOCK(t->lock);
            handler(t, data);
            CDBUS_LOCK(t->lock);
        }
        else
        {
            CDBUS_TRACE((CDBUS_TRC_INFO, "No timeout handler configured"));
        }

        /* Reschedule the timer (or not) based on the repeat mode */
        if ( t->enabled && t->repeat )
        {
            /* If it's not currently active then ... */
            if ( !loop->timerIsEnabled(timer) )
            {

                loop->timerStart(timer);
                cdbus_dispatcherWakeup(t->dispatcher);
            }
        }
        else
        {
            /* If it's currently active then ... */
            if ( loop->timerIsEnabled(timer) )
            {
                loop->timerStop(timer);
                cdbus_dispatcherWakeup(t->dispatcher);
            }
        }

        CDBUS_UNLOCK(t->lock);
        cdbus_timeoutUnref(t);
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

    if ( NULL != dispatcher )
    {
        timeout = cdbus_calloc(1, sizeof(*timeout));
        if ( NULL != timeout )
        {
            CDBUS_LOCK_ALLOC(timeout->lock, CDBUS_MUTEX_RECURSIVE);
            if ( CDBUS_LOCK_IS_NULL(timeout->lock) )
            {
                cdbus_free(timeout);
                timeout = NULL;
            }
            else
            {
                timeout->timer = dispatcher->loop->timerNew(dispatcher->loop,
                                            (cdbus_UInt32)msecInterval,
                                            cdbus_timerCallback,
                                            timeout);
                if ( NULL == timeout->timer )
                {
                    CDBUS_LOCK_FREE(timeout->lock);
                    cdbus_free(timeout);
                    timeout = NULL;
                }
                else
                {
                    timeout->dispatcher = cdbus_dispatcherRef(dispatcher);
                    timeout->data = data;
                    timeout->handler = h;
                    timeout->repeat = repeat;
                    timeout->enabled = CDBUS_FALSE;
                    timeout = cdbus_timeoutRef(timeout);
                    CDBUS_TRACE((CDBUS_TRC_INFO,
                          "Created timeout instance (%p)", (void*)timeout));
                }
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
            if (  t->dispatcher->loop->timerIsEnabled(t->timer) )
            {
                t->dispatcher->loop->timerStop(t->timer);
            }
            t->dispatcher->loop->timerDestroy(t->timer);
            cdbus_dispatcherUnref(t->dispatcher);
            CDBUS_UNLOCK(t->lock);
            CDBUS_LOCK_FREE(t->lock);
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
        isEnabled = t->enabled;
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

        /* If we want to enable the timeout then ... */
        if ( option )
        {
            /* If it's not currently active then ... */
            if ( !t->dispatcher->loop->timerIsEnabled(t->timer) )
            {
                t->dispatcher->loop->timerStart(t->timer);
                cdbus_dispatcherWakeup(t->dispatcher);
            }
        }
        /* Else disable the watcher */
        else
        {
            /* If the watch is currently active then */
            if ( t->dispatcher->loop->timerIsEnabled(t->timer) )
            {
                t->dispatcher->loop->timerStop(t->timer);
                cdbus_dispatcherWakeup(t->dispatcher);
            }
        }

        /* Update the new state */
        t->enabled = option;

        CDBUS_UNLOCK(t->lock);
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
        period = t->dispatcher->loop->timerGetInterval(t->timer);
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
        t->dispatcher->loop->timerSetInterval(t->timer,
                                            (cdbus_UInt32)msecInterval);
        CDBUS_UNLOCK(t->lock);
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
        /* If repeat mode is being enabled then ... */
        if ( repeat )
        {
            /* If repeat was previously disabled but the timeout is still
             * enabled then ...
             * */
            if ( !t->repeat && t->enabled )
            {
                /* User wants to repeat the timeout and the timeout is enabled
                 * so we'll re-start the timeout */
                t->dispatcher->loop->timerStart(t->timer);
                cdbus_dispatcherWakeup(t->dispatcher);
            }
        }
        t->repeat = repeat;
        CDBUS_UNLOCK(t->lock);
    }
}

#endif /* Guard for CDBUS_TIMEOUT_PRIV_C_ */
