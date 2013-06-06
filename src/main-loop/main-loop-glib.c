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
 * @file           main-loop-glib.c
 * @author         Glenn Schmottlach
 * @brief          Implementation of GLIB main loop.
 *===========================================================================
 */
#include "cdbus/cdbus.h"
#include "dbus/dbus.h"
#include "cdbus/main-loop-glib.h"
#include <assert.h>
#include <string.h>


#define CDBUS_MAINLOOP_DEFAULT_LOOP     (NULL)


struct cdbus_MainLoopTimer
{
    GSource*                    timerSrc;
    cdbus_Bool                  isActive;
    guint                       msecInterval;
    cdbus_MainLoopTimerCbFunc   handler;
    cdbus_MainLoopGlib*         loop;
    void*                       userData;
};

struct cdbus_MainLoopWatch
{
    GSource*                    ioSrc;
    cdbus_Bool                  isActive;
    cdbus_Descriptor            fd;
    gushort                     flags;
    cdbus_MainLoopWatchCbFunc   handler;
    cdbus_MainLoopGlib*         loop;
    void*                       userData;
};


static cdbus_MainLoop*
cdbus_mainLoopRef
    (
    cdbus_MainLoop* loop
    )
{
    cdbus_MainLoopGlib* self = (cdbus_MainLoopGlib*)loop;
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
    cdbus_MainLoopGlib* self = (cdbus_MainLoopGlib*)loop;

    if ( self != NULL )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&self->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            cdbus_mainLoopGlibFree(cdbus_mainLoopGlibDestroy(self));
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
    GMainContext* ctx;
    cdbus_MainLoopGlib* self = (cdbus_MainLoopGlib*)loop;

    if ( self != NULL )
    {
        if ( self->vtable.loopPre != NULL )
        {
            self->vtable.loopPre(loop);
        }

        ctx = g_main_loop_get_context(self->glibLoop);
        g_main_context_iteration(ctx, canBlock);

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
    cdbus_MainLoopGlib* self = (cdbus_MainLoopGlib*)loop;

    if ( self != NULL )
    {
        g_main_loop_quit(self->glibLoop);
    }
}


static void
cdbus_timerDestroyNotify
    (
    gpointer data
    )
{
    cdbus_MainLoopTimer* timer;
    if ( NULL != data )
    {
        timer = (cdbus_MainLoopTimer*)data;
        g_source_unref(timer->timerSrc);
        timer->timerSrc = NULL;
    }
}


static gboolean
cdbus_timerCallback
    (
    gpointer t
    )
{
    cdbus_MainLoopTimer* timer;
    if ( NULL != t )
    {
        timer = (cdbus_MainLoopTimer*)(t);
        timer->handler(timer, timer->userData);
    }

    /* Leave timer enabled so it will go off again */
    return TRUE;
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
    cdbus_MainLoopGlib* self = (cdbus_MainLoopGlib*)loop;
    cdbus_MainLoopTimer* timer = NULL;

    if ( self != NULL )
    {
        timer = (cdbus_MainLoopTimer*)cdbus_calloc(1, sizeof(*timer));
        if ( NULL != timer )
        {
            timer->timerSrc = NULL;
            timer->isActive = CDBUS_FALSE;
            timer->msecInterval = msecTimeout;
            timer->handler = handler;
            cdbus_mainLoopRef(loop);
            timer->loop = self;
            timer->userData = userData;
        }
    }

    return timer;
}

static void
cdbus_mainLoopTimerStop
    (
    cdbus_MainLoopTimer*    timer
    )
{
    if ( NULL != timer )
    {
        if ( timer->isActive )
        {
            g_source_destroy(timer->timerSrc);
            timer->isActive = CDBUS_FALSE;
        }
    }
}


static void
cdbus_mainLoopTimerDestroy
    (
    cdbus_MainLoopTimer*    timer
    )
{
    cdbus_MainLoopGlib* self;
    if ( NULL != timer )
    {
        assert(NULL != timer->loop);
        self = timer->loop;
        cdbus_mainLoopTimerStop(timer);
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
        isEnabled = timer->isActive;
    }
    return isEnabled;
}


static void
cdbus_mainLoopTimerStart
    (
    cdbus_MainLoopTimer*    timer
    )
{
    GMainContext* ctx;

    if ( NULL != timer )
    {
        /* Stop it in case it was already running */
        cdbus_mainLoopTimerStop(timer);

        timer->timerSrc = g_timeout_source_new(timer->msecInterval);
        g_source_set_callback(timer->timerSrc, cdbus_timerCallback,
                                timer, cdbus_timerDestroyNotify);
        ctx = g_main_loop_get_context(timer->loop->glibLoop);
        g_source_attach(timer->timerSrc, ctx);
        timer->isActive = CDBUS_TRUE;
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
        timer->msecInterval = msec;
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
        msecPeriod = timer->msecInterval;
    }

    return msecPeriod;
}


static gushort
cdbus_convertToGlibFlags
    (
    cdbus_UInt32 dbusFlags
    )
{
    gushort gFlags = 0U;

    if ( dbusFlags & DBUS_WATCH_READABLE ) gFlags |= G_IO_IN;
    if ( dbusFlags & DBUS_WATCH_WRITABLE ) gFlags |= G_IO_OUT;

    if ( dbusFlags & DBUS_WATCH_ERROR )
    {
        gFlags |= G_IO_ERR;
    }

    if ( dbusFlags & DBUS_WATCH_HANGUP )
    {
        gFlags |= G_IO_HUP | G_IO_ERR;
    }

    return gFlags;
}


static cdbus_UInt32
cdbus_convertToDbusFlags
    (
    gushort gFlags
    )
{
    cdbus_UInt32 dbusFlags = 0U;

    if ( gFlags & G_IO_IN ) dbusFlags |= DBUS_WATCH_READABLE;
    if ( gFlags & G_IO_OUT ) dbusFlags |= DBUS_WATCH_WRITABLE;
    if ( gFlags & G_IO_ERR ) dbusFlags |= DBUS_WATCH_ERROR;
    if ( gFlags & G_IO_HUP ) dbusFlags |= DBUS_WATCH_HANGUP;

    return dbusFlags;
}


static void
cdbus_watchDestroyNotify
    (
    gpointer data
    )
{
    cdbus_MainLoopWatch* watch;
    if ( NULL != data )
    {
        watch = (cdbus_MainLoopWatch*)data;
        g_source_unref(watch->ioSrc);
        watch->ioSrc = NULL;
    }
}


static gboolean
cdbus_watchCallback
    (
    GIOChannel*     source,
    GIOCondition    condition,
    gpointer        data
    )
{
    cdbus_MainLoopWatch* watch;

    CDBUS_UNUSED(source);

    if ( NULL != data )
    {
        watch = (cdbus_MainLoopWatch*)data;
        watch->handler(watch, cdbus_convertToDbusFlags(condition),
                        watch->userData);
    }

    /* Return TRUE to keep it enabled */
    return TRUE;
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
    cdbus_MainLoopGlib* self = (cdbus_MainLoopGlib*)loop;
    cdbus_MainLoopWatch* watch = NULL;

    if ( self != NULL )
    {
        watch = (cdbus_MainLoopWatch*)cdbus_calloc(1, sizeof(*watch));
        if ( NULL != watch )
        {
            watch->ioSrc = NULL;
            watch->isActive = CDBUS_FALSE;
            watch->fd = fd;
            watch->flags = cdbus_convertToGlibFlags(flags);
            watch->handler = handler;
            cdbus_mainLoopRef(loop);
            watch->loop = self;
            watch->userData = userData;
        }
    }

    return watch;
}


static void
cdbus_mainLoopWatchStop
    (
    cdbus_MainLoopWatch*    watch
    )
{
    if ( NULL != watch )
    {
        if ( watch->isActive )
        {
            g_source_destroy(watch->ioSrc);
            watch->isActive = FALSE;
        }
    }
}


static void
cdbus_mainLoopWatchDestroy
    (
    cdbus_MainLoopWatch*    watch
    )
{
    cdbus_MainLoopGlib* self;
    if ( NULL != watch )
    {
        assert(NULL != watch->loop);
        self = watch->loop;
        cdbus_mainLoopWatchStop(watch);
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
        isEnabled = watch->isActive;
    }
    return isEnabled;
}


static void
cdbus_mainLoopWatchStart
    (
    cdbus_MainLoopWatch*    watch
    )
{
    GMainContext* ctx;
    GIOChannel*   chan;

    if ( NULL != watch )
    {
        /* Stop it in case it was already running */
        cdbus_mainLoopWatchStop(watch);

        chan = g_io_channel_unix_new(watch->fd);
        watch->ioSrc = g_io_create_watch(chan, watch->flags);
        g_io_channel_unref(chan);
        g_source_set_callback(watch->ioSrc, (GSourceFunc)cdbus_watchCallback,
                                watch, cdbus_watchDestroyNotify);
        ctx = g_main_loop_get_context(watch->loop->glibLoop);
        g_source_attach(watch->ioSrc, ctx);
        watch->isActive = CDBUS_TRUE;
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
        fd = watch->fd;
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
        watch->flags = cdbus_convertToGlibFlags(flags);
        if ( cdbus_mainLoopWatchIsEnabled(watch) )
        {
            cdbus_mainLoopWatchStart(watch);
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
        flags = cdbus_convertToDbusFlags(watch->flags);
    }
    return flags;
}


cdbus_MainLoopGlib*
cdbus_mainLoopGlibAlloc()
{
    return (cdbus_MainLoopGlib*)cdbus_calloc(1, sizeof(cdbus_MainLoopGlib));
}


cdbus_MainLoopGlib*
cdbus_mainLoopGlibNew
    (
    cdbus_MainLoopGlib* self,
    GMainLoop*          loop,
    cdbus_Bool          ownsLoop,
    void*               userData
    )
{
    if ( NULL != self )
    {

        if ( NULL != loop )
        {
            self->glibLoop = loop;
            self->ownsLoop = ownsLoop;
            g_main_loop_ref(self->glibLoop);
        }
        else
        {
            /* It is assumed this implicitly adds a reference to the loop */
            self->glibLoop = g_main_loop_new(CDBUS_MAINLOOP_DEFAULT_LOOP, FALSE);
            self->ownsLoop = CDBUS_FALSE;
        }

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


cdbus_MainLoopGlib*
cdbus_mainLoopGlibDestroy
    (
    cdbus_MainLoopGlib*   self
    )
{
    if ( NULL != self )
    {
        g_main_loop_unref(self->glibLoop);
        memset(self, 0, sizeof(*self));
    }

    return self;
}


void
cdbus_mainLoopGlibFree
    (
    cdbus_MainLoopGlib*   self
    )
{
    cdbus_free(self);
}


