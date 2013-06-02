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
 * @file           mainloop.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of a main loop abstraction.
 *===========================================================================
 */

#ifndef CDBUS_MAINLOOP_H_
#define CDBUS_MAINLOOP_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
typedef struct cdbus_MainLoop cdbus_MainLoop;
typedef struct cdbus_MainLoopTimer cdbus_MainLoopTimer;
typedef struct cdbus_MainLoopWatch cdbus_MainLoopWatch;

/* Basic main loop function prototypes */
typedef cdbus_MainLoop* (*cdbus_MainLoopRefFunc)(cdbus_MainLoop* loop);
typedef void (*cdbus_MainLoopUnrefFunc)(cdbus_MainLoop* loop);
typedef void (*cdbus_MainLoopIterateFunc)(cdbus_MainLoop* loop,
                                            cdbus_Bool canBlock);
typedef void (*cdbus_MainLoopQuitFunc)(cdbus_MainLoop* loop);
typedef void (*cdbus_MainLoopPreLoopFunc)(cdbus_MainLoop* loop);
typedef void (*cdbus_MainLoopPostLoopFunc)(cdbus_MainLoop* loop);

/* Main loop timer function prototypes */
typedef void (*cdbus_MainLoopTimerCbFunc)(cdbus_MainLoopTimer* timer,
                                            void* userData);
typedef cdbus_MainLoopTimer* (*cdbus_MainLoopTimerNewFunc)(cdbus_MainLoop* loop,
                                            cdbus_UInt32 msecTimeout,
                                            cdbus_MainLoopTimerCbFunc handler,
                                            void* userData);
typedef void (*cdbus_MainLoopTimerDestroyFunc)(cdbus_MainLoopTimer* timer);
typedef cdbus_Bool (*cdbus_MainLoopTimerIsEnabledFunc)(cdbus_MainLoopTimer* timer);
typedef void (*cdbus_MainLoopTimerStartFunc)(cdbus_MainLoopTimer* timer);
typedef void (*cdbus_MainLoopTimerStopFunc)(cdbus_MainLoopTimer* timer);
typedef void (*cdbus_MainLoopTimerSetIntervalFunc)(cdbus_MainLoopTimer* timer, cdbus_UInt32 msec);
typedef cdbus_UInt32(*cdbus_MainLoopTimerGetIntervalFunc)(cdbus_MainLoopTimer* timer);

/* Main loop watch function prototypes */
typedef void (*cdbus_MainLoopWatchCbFunc)(cdbus_MainLoopWatch* watch,
                                            cdbus_UInt32 flags, void* userData);
typedef cdbus_MainLoopWatch* (*cdbus_MainLoopWatchNewFunc)(cdbus_MainLoop* loop,
                                                            cdbus_Descriptor fd,
                                                            cdbus_UInt32 flags,
                                                            cdbus_MainLoopWatchCbFunc handler,
                                                            void* userData);
typedef void (*cdbus_MainLoopWatchDestroyFunc)(cdbus_MainLoopWatch* watch);
typedef cdbus_Bool (*cdbus_MainLoopWatchIsEnabledFunc)(cdbus_MainLoopWatch* watch);
typedef void (*cdbus_MainLoopWatchStartFunc)(cdbus_MainLoopWatch* watch);
typedef void (*cdbus_MainLoopWatchStopFunc)(cdbus_MainLoopWatch* watch);
typedef cdbus_Descriptor (*cdbus_MainLoopWatchGetDescriptorFunc)(cdbus_MainLoopWatch* watch);
typedef void (*cdbus_MainLoopWatchSetFlagsFunc)(cdbus_MainLoopWatch* watch, cdbus_UInt32 flags);
typedef cdbus_UInt32 (*cdbus_MainLoopWatchGetFlagsFunc)(cdbus_MainLoopWatch* watch);

struct cdbus_MainLoop
{
    cdbus_MainLoopRefFunc                   loopRef;
    cdbus_MainLoopUnrefFunc                 loopUnref;
    cdbus_MainLoopIterateFunc               loopIterate;
    cdbus_MainLoopQuitFunc                  loopQuit;
    cdbus_MainLoopPreLoopFunc               loopPre;
    cdbus_MainLoopPostLoopFunc              loopPost;

    cdbus_MainLoopTimerNewFunc              timerNew;
    cdbus_MainLoopTimerDestroyFunc          timerDestroy;
    cdbus_MainLoopTimerStartFunc            timerStart;
    cdbus_MainLoopTimerStopFunc             timerStop;
    cdbus_MainLoopTimerIsEnabledFunc        timerIsEnabled;
    cdbus_MainLoopTimerSetIntervalFunc      timerSetInterval;
    cdbus_MainLoopTimerGetIntervalFunc      timerGetInterval;

    cdbus_MainLoopWatchNewFunc              watchNew;
    cdbus_MainLoopWatchDestroyFunc          watchDestroy;
    cdbus_MainLoopWatchStartFunc            watchStart;
    cdbus_MainLoopWatchStopFunc             watchStop;
    cdbus_MainLoopWatchIsEnabledFunc        watchIsEnabled;
    cdbus_MainLoopWatchGetDescriptorFunc    watchGetDescriptor;
    cdbus_MainLoopWatchSetFlagsFunc         watchSetFlags;
    cdbus_MainLoopWatchGetFlagsFunc         watchGetFlags;
};

#define CDBUS_MAIN_LOOP_DECLARE cdbus_MainLoop vtable



CDBUS_END_DECLS


#endif /* Guard for CDBUS_MAINLOOP_H_ */
