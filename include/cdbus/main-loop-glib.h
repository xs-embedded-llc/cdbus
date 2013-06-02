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
 * @file           main-loop-glib.h
 * @author         Glenn Schmottlach
 * @brief          Declarations of the public glib based main loop.
 *===========================================================================
 */

#ifndef CDBUS_MAIN_LOOP_GLIB_H_
#define CDBUS_MAIN_LOOP_GLIB_H_

#include "cdbus/mainloop.h"
#include "glib.h"


typedef struct cdbus_MainLoopGlib
{
    /* This **MUST** be declared first in any derived loop structure */
    CDBUS_MAIN_LOOP_DECLARE;

    GMainLoop*      glibLoop;
    cdbus_Bool      ownsLoop;
    void*           userData;
    cdbus_Atomic    refCnt;
} cdbus_MainLoopGlib;

/*
 * The following functions are needed to support C-style "inheritance"
 * of the base glib main loop. Clients typically don't need to
 * access these directly but instead should utilize the macros
 * that follow.
 */

CDBUS_EXPORT cdbus_MainLoopGlib* cdbus_mainLoopGlibAlloc();
CDBUS_EXPORT cdbus_MainLoopGlib* cdbus_mainLoopGlibNew(cdbus_MainLoopGlib* self,
                                                    GMainLoop* loop,
                                                    cdbus_Bool ownsLoop,
                                                    void* userData);
CDBUS_EXPORT cdbus_MainLoopGlib* cdbus_mainLoopGlibDestroy(
                                                    cdbus_MainLoopGlib* self);
CDBUS_EXPORT void cdbus_mainLoopGlibFree(cdbus_MainLoopGlib* self);


/*
 * Client API for creating/referencing/unreferencing glib based main loops.
 */
#define CDBUS_MAIN_LOOP_GLIB_NEW(glibLoop, ownsLoop, userdata) \
    ((cdbus_MainLoop*)cdbus_mainLoopGlibNew(cdbus_mainLoopGlibAlloc(), \
                                        glibLoop, \
                                        ownsLoop, \
                                        userdata))

#define CDBUS_MAIN_LOOP_GLIB_REF(loop) \
    ((cdbus_MainLoop*)((cdbus_MainLoopGlib*)loop)->vtable.loopRef((cdbus_MainLoop*)loop))

#define CDBUS_MAIN_LOOP_GLIB_UNREF(loop) \
    ((cdbus_MainLoopGlib*)loop)->vtable.loopUnref((cdbus_MainLoop*)loop)


#endif /* Guard for CDBUS_MAIN_LOOP_GLIB_H_ */
