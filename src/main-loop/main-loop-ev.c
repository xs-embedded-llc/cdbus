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
 * @file           main-loop-ev.c
 * @author         Glenn Schmottlach
 * @brief          Implementation of libev main loop.
 *===========================================================================
 */
#include "cdbus/cdbus.h"
#include "dbus/dbus.h"
#include "cdbus/main-loop-ev.h"
#include "ev.h"
#include <assert.h>
#include <string.h>


#define CDBUS_MAINLOOP_DEFAULT_LOOP   ((void*)ev_default_loop(0))

#if EV_MULTIPLICITY
#define CDBUS_MAINLOOP_EV self->loop
#define CDBUS_MAINLOOP_EV_ CDBUS_MAINLOOP_EV,
#define CDBUS_EV_UNUSED(X) CDBUS_UNUSED(X)
#else
#define CDBUS_MAINLOOP_EV
#define CDBUS_MAINLOOP_EV_
#define CDBUS_EV_UNUSED(X)
#endif


struct cdbus_MainLoopTimer
{
    ev_timer                    evTimer;
    cdbus_MainLoopTimerCbFunc   handler;
    cdbus_MainLoopEv*           loop;
    void*                       userData;
};

struct cdbus_MainLoopWatch
{
    ev_io                       evIo;
    cdbus_MainLoopWatchCbFunc   handler;
    cdbus_MainLoopEv*           loop;
    void*                       userData;
};


static cdbus_MainLoop*
cdbus_mainLoopRef
    (
    cdbus_MainLoop* loop
    )
{
    cdbus_MainLoopEv* self = (cdbus_MainLoopEv*)loop;
    if ( NULL != loop )
    {
        cdbus_atomicAdd(&self->refCnt, 1);
    }

    return loop;
}


static void
cdbus_mainLoopUnref
    (
    cdbus_MainLoop* loop
    )
{
    cdbus_Int32 value = 0;
    cdbus_MainLoopEv* self = (cdbus_MainLoopEv*)loop;

    if ( self != NULL )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&self->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            cdbus_mainLoopEvFree(cdbus_mainLoopEvDestroy(self));
        }
    }
}


static void
cdbus_mainLoopIterate
    (
    cdbus_MainLoop* loop,
    cdbus_Bool      canBlock
    )
{
    cdbus_MainLoopEv* self = (cdbus_MainLoopEv*)loop;

    if ( self != NULL )
    {
        if ( self->vtable.loopPre != NULL )
        {
            self->vtable.loopPre(loop);
        }

        ev_run(CDBUS_MAINLOOP_EV_ canBlock ? EVRUN_ONCE : EVRUN_NOWAIT);

        if ( self->vtable.loopPost != NULL )
        {
            self->vtable.loopPost(loop);
        }
    }
}


static void
cdbus_mainLoopQuit
    (
    cdbus_MainLoop* loop
    )
{
    cdbus_MainLoopEv* self = (cdbus_MainLoopEv*)loop;

    if ( self != NULL )
    {
        ev_break(CDBUS_MAINLOOP_EV_ EVBREAK_ALL);
    }
}


static void
cdbus_timerCallback
    (
    EV_P_
    ev_timer*   t,
    int         events
    )
{
    CDBUS_EV_UNUSED(EV_A);
    CDBUS_UNUSED(events);

    cdbus_MainLoopTimer* timer;
    if ( (NULL != t) && (NULL != t->data) )
    {
        timer = (cdbus_MainLoopTimer*)(t->data);
        timer->handler(timer, timer->userData);
    }
}


static cdbus_MainLoopTimer*
cdbus_mainLoopTimerNew
    (
    cdbus_MainLoop*             loop,
    cdbus_UInt32                msecTimeout,
    cdbus_MainLoopTimerCbFunc   handler,
    void*                       userData
    )
{
    cdbus_MainLoopEv* self = (cdbus_MainLoopEv*)loop;
    cdbus_MainLoopTimer* timer = NULL;

    if ( self != NULL )
    {
        timer = (cdbus_MainLoopTimer*)cdbus_calloc(1, sizeof(*timer));
        if ( NULL != timer )
        {
            timer->evTimer.data = timer;
            ev_timer_init(&timer->evTimer, cdbus_timerCallback,
                            0.0, (ev_tstamp)msecTimeout / 1000.0);
            timer->handler = handler;
            cdbus_mainLoopRef(loop);
            timer->loop = self;
            timer->userData = userData;
        }
    }

    return timer;
}


static void
cdbus_mainLoopTimerDestroy
    (
    cdbus_MainLoopTimer*    timer
    )
{
    cdbus_MainLoopEv* self;
    if ( NULL != timer )
    {
        assert(NULL != timer->loop);
        self = timer->loop;
        ev_timer_stop(CDBUS_MAINLOOP_EV_  &timer->evTimer);
        cdbus_mainLoopUnref(&self->vtable);
        cdbus_free(timer);
    }
}


static cdbus_Bool
cdbus_mainLoopTimerIsEnabled
    (
    cdbus_MainLoopTimer*    timer
    )
{
    cdbus_Bool isEnabled = CDBUS_FALSE;

    if ( NULL != timer )
    {
        isEnabled = ev_is_active(&(timer->evTimer)) ? CDBUS_TRUE : CDBUS_FALSE;
    }
    return isEnabled;
}


static void
cdbus_mainLoopTimerStart
    (
    cdbus_MainLoopTimer*    timer
    )
{
    cdbus_MainLoopEv* self;
    if ( NULL != timer )
    {
        assert(NULL != timer->loop);
        self = timer->loop;
        ev_timer_again(CDBUS_MAINLOOP_EV_ &timer->evTimer);
    }
}


static void
cdbus_mainLoopTimerStop
    (
    cdbus_MainLoopTimer*    timer
    )
{
    cdbus_MainLoopEv* self;
    if ( NULL != timer )
    {
        assert(NULL != timer->loop);
        self = timer->loop;
        ev_timer_stop(CDBUS_MAINLOOP_EV_ &timer->evTimer);
    }
}


static void
cdbus_mainLoopTimerSetInterval
    (
    cdbus_MainLoopTimer*    timer,
    cdbus_UInt32            msec
    )
{
    if ( NULL != timer )
    {
        /* This will take effect the *next* time the timer is restarted */
        timer->evTimer.repeat = (ev_tstamp)msec / 1000.0;
    }
}


static cdbus_UInt32
cdbus_mainLoopTimerGetInterval
    (
    cdbus_MainLoopTimer*    timer
    )
{
    cdbus_UInt32 msecPeriod = 0;

    if ( NULL != timer )
    {
        msecPeriod = (cdbus_UInt32)(timer->evTimer.repeat * 1000);
    }

    return msecPeriod;
}


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
        evFlags |= EV_ERROR;
    }

    if ( dbusFlags & DBUS_WATCH_HANGUP )
    {
        evFlags |= EV_CUSTOM | EV_ERROR;
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
    if ( evFlags & EV_ERROR ) dbusFlags |= DBUS_WATCH_ERROR;
    if ( evFlags & EV_CUSTOM ) dbusFlags |= DBUS_WATCH_HANGUP;

    return dbusFlags;
}

static void
cdbus_watchCallback
    (
    EV_P_
    ev_io*      w,
    int         events
    )
{
    cdbus_MainLoopWatch* watch;

    CDBUS_EV_UNUSED(EV_A);

    if ( (NULL != w) && (NULL != w->data) )
    {
        watch = (cdbus_MainLoopWatch*)(w->data);
        watch->handler(watch, cdbus_convertToDbusFlags(events), watch->userData);
    }
}


static cdbus_MainLoopWatch*
cdbus_mainLoopWatchNew
    (
    cdbus_MainLoop*             loop,
    cdbus_Descriptor            fd,
    cdbus_UInt32                flags,
    cdbus_MainLoopWatchCbFunc   handler,
    void*                       userData
    )
{
    cdbus_MainLoopEv* self = (cdbus_MainLoopEv*)loop;
    cdbus_MainLoopWatch* watch = NULL;

    if ( self != NULL )
    {
        watch = (cdbus_MainLoopWatch*)cdbus_calloc(1, sizeof(*watch));
        if ( NULL != watch )
        {
            ev_io_init(&watch->evIo, cdbus_watchCallback, fd,
                        cdbus_convertToEvFlags(flags));
            watch->evIo.data = watch;
            watch->handler = handler;
            cdbus_mainLoopRef(loop);
            watch->loop = self;
            watch->userData = userData;
        }
    }

    return watch;
}


static void
cdbus_mainLoopWatchDestroy
    (
    cdbus_MainLoopWatch*    watch
    )
{
    cdbus_MainLoopEv* self;
    if ( NULL != watch )
    {
        assert(NULL != watch->loop);
        self = watch->loop;
        ev_io_stop(CDBUS_MAINLOOP_EV_  &watch->evIo);
        cdbus_mainLoopUnref(&self->vtable);
        cdbus_free(watch);
    }
}


static cdbus_Bool
cdbus_mainLoopWatchIsEnabled
    (
    cdbus_MainLoopWatch*    watch
    )
{
    cdbus_Bool isEnabled = CDBUS_FALSE;

    if ( NULL != watch )
    {
        isEnabled = ev_is_active(&(watch->evIo)) ? CDBUS_TRUE : CDBUS_FALSE;
    }
    return isEnabled;
}


static void
cdbus_mainLoopWatchStart
    (
    cdbus_MainLoopWatch*    watch
    )
{
    cdbus_MainLoopEv* self;
    if ( NULL != watch )
    {
        assert(NULL != watch->loop);
        self = watch->loop;
        ev_io_start(CDBUS_MAINLOOP_EV_ &watch->evIo);
    }
}


static void
cdbus_mainLoopWatchStop
    (
    cdbus_MainLoopWatch*    watch
    )
{
    cdbus_MainLoopEv* self;
    if ( NULL != watch )
    {
        assert(NULL != watch->loop);
        self = watch->loop;
        ev_io_stop(CDBUS_MAINLOOP_EV_ &watch->evIo);
    }
}


static cdbus_Descriptor
cdbus_mainLoopWatchGetDescriptor
    (
    cdbus_MainLoopWatch*    watch
    )
{
    cdbus_Descriptor fd = -1;

    if ( NULL != watch )
    {
        fd = watch->evIo.fd;
    }
    return fd;
}


static void
cdbus_mainLoopWatchSetFlags
    (
    cdbus_MainLoopWatch*    watch,
    cdbus_UInt32            flags
    )
{
    if ( NULL != watch )
    {
        if ( cdbus_mainLoopWatchIsEnabled(watch) )
        {
            cdbus_mainLoopWatchStop(watch);
            ev_io_set(&watch->evIo,  watch->evIo.fd,
                      cdbus_convertToEvFlags(flags));
            cdbus_mainLoopWatchStart(watch);
        }
        else
        {
            ev_io_set(&watch->evIo,  watch->evIo.fd,
                      cdbus_convertToEvFlags(flags));
        }
    }
}


static cdbus_UInt32
cdbus_mainLoopWatchGetFlags
    (
    cdbus_MainLoopWatch*    watch
    )
{
    cdbus_UInt32 flags = 0U;

    if ( NULL != watch )
    {
        flags = cdbus_convertToDbusFlags(watch->evIo.events);
    }
    return flags;
}


cdbus_MainLoopEv*
cdbus_mainLoopEvAlloc()
{
    return (cdbus_MainLoopEv*)cdbus_calloc(1, sizeof(cdbus_MainLoopEv));
}


cdbus_MainLoopEv*
cdbus_mainLoopEvNew
    (
    cdbus_MainLoopEv*   self,
    EV_P_
    cdbus_Bool          ownsLoop,
    void*               userData
    )
{
    if ( NULL != self )
    {
#if EV_MULTIPLICITY
        if ( NULL != loop )
        {
            self->loop = loop;
            self->ownsLoop = ownsLoop;
        }
        else
        {
            self->loop = CDBUS_MAINLOOP_DEFAULT_LOOP;
            self->ownsLoop = CDBUS_FALSE;
        }
#else
        self->ownsLoop = CDBUS_FALSE;
#endif
        self->userData = NULL;
        self->refCnt.value = 1;
        self->userData = userData;

        self->vtable.loopRef = cdbus_mainLoopRef;
        self->vtable.loopUnref = cdbus_mainLoopUnref;
        self->vtable.loopIterate = cdbus_mainLoopIterate;
        self->vtable.loopQuit = cdbus_mainLoopQuit;
        self->vtable.loopPre = NULL;
        self->vtable.loopPost = NULL;

        self->vtable.timerNew = cdbus_mainLoopTimerNew;
        self->vtable.timerDestroy = cdbus_mainLoopTimerDestroy;
        self->vtable.timerIsEnabled = cdbus_mainLoopTimerIsEnabled;
        self->vtable.timerStart = cdbus_mainLoopTimerStart;
        self->vtable.timerStop = cdbus_mainLoopTimerStop;
        self->vtable.timerSetInterval = cdbus_mainLoopTimerSetInterval;
        self->vtable.timerGetInterval = cdbus_mainLoopTimerGetInterval;

        self->vtable.watchNew = cdbus_mainLoopWatchNew;
        self->vtable.watchDestroy = cdbus_mainLoopWatchDestroy;
        self->vtable.watchIsEnabled = cdbus_mainLoopWatchIsEnabled;
        self->vtable.watchStart = cdbus_mainLoopWatchStart;
        self->vtable.watchStop = cdbus_mainLoopWatchStop;
        self->vtable.watchGetDescriptor = cdbus_mainLoopWatchGetDescriptor;
        self->vtable.watchSetFlags = cdbus_mainLoopWatchSetFlags;
        self->vtable.watchGetFlags = cdbus_mainLoopWatchGetFlags;
    }

    return self;
}


cdbus_MainLoopEv*
cdbus_mainLoopEvDestroy
    (
    cdbus_MainLoopEv*   self
    )
{
    if ( NULL != self )
    {
#if EV_MULTIPLICITY
        if ( self->ownsLoop )
        {
            self->vtable.loopQuit((cdbus_MainLoop*)self);
            ev_loop_destroy(self->loop);
        }
#endif
        memset(self, 0, sizeof(*self));
    }

    return self;
}


void
cdbus_mainLoopEvFree
    (
    cdbus_MainLoopEv*   self
    )
{
    cdbus_free(self);
}


