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
 * @file           main-loop-ev.h
 * @author         Glenn Schmottlach
 * @brief          Declarations of the public libev based main loop.
 *===========================================================================
 */

#ifndef CDBUS_MAIN_LOOP_EV_H_
#define CDBUS_MAIN_LOOP_EV_H_

#include "cdbus/mainloop.h"
#include "ev.h"


typedef struct cdbus_MainLoopEv
{
    /* This **MUST** be declared first in any derived loop structure */
    CDBUS_MAIN_LOOP_DECLARE;

    EV_P;
    cdbus_Bool      ownsLoop;
    void*           userData;
    cdbus_Atomic    refCnt;
} cdbus_MainLoopEv;

/*
 * The following functions are needed to support C-style "inheritance"
 * of the base libev main loop. Clients typically don't need to
 * access these directly but instead should utilize the macros
 * that follow.
 */
CDBUS_EXPORT cdbus_MainLoopEv* cdbus_mainLoopEvAlloc();
CDBUS_EXPORT cdbus_MainLoopEv* cdbus_mainLoopEvNew(cdbus_MainLoopEv* self,
                                                    EV_P_
                                                    cdbus_Bool ownsLoop,
                                                    void* userData);
CDBUS_EXPORT cdbus_MainLoopEv* cdbus_mainLoopEvDestroy(cdbus_MainLoopEv* self);
CDBUS_EXPORT void cdbus_mainLoopEvFree(cdbus_MainLoopEv* self);

/*
 * Client API for creating/referencing/unreferencing libev based main loops.
 */
#if EV_MULTIPLICITY
#define CDBUS_MAIN_LOOP_EV_NEW(evLoop, ownsLoop, userdata) \
    ((cdbus_MainLoop*)cdbus_mainLoopEvNew(cdbus_mainLoopEvAlloc(), \
                                        evLoop, \
                                        ownsLoop, \
                                        userdata))
#else
#define CDBUS_MAIN_LOOP_EV_NEW(ownsLoop, userdata) \
    ((cdbus_MainLoop*)cdbus_mainLoopEvNew(cdbus_mainLoopEvAlloc(), \
                                        ownsLoop, \
                                        userdata))
#endif

#define CDBUS_MAIN_LOOP_EV_REF(loop) \
    ((cdbus_MainLoop*)((cdbus_MainLoopEv*)loop)->vtable.loopRef((cdbus_MainLoop*)loop))

#define CDBUS_MAIN_LOOP_EV_UNREF(loop) \
    ((cdbus_MainLoopEv*)loop)->vtable.loopUnref((cdbus_MainLoop*)loop)


#endif /* Guard for CDBUS_MAIN_LOOP_EV_H_ */
