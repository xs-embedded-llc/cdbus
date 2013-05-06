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
 * @file           dispatcher.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of D-Bus dispatcher class.
 *===========================================================================
 */

#ifndef CDBUS_DISPATCHER_H_
#define CDBUS_DISPATCHER_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"
#include "ev.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
typedef struct cdbus_Dispatcher cdbus_Dispatcher;
struct cdbus_Connection;
struct cdbus_Watch;
struct cdbus_Timeout;

#define CDBUS_DISPATCHER_A  dispatcher
#define CDBUS_DISPATCHER_A_ CDBUS_DISPATCHER_A,
#define CDBUS_DISPATCHER_P  cdbus_Dispatcher* CDBUS_DISPATCHER_A
#define CDBUS_DISPATCHER_P_  CDBUS_DISPATCHER_P,

typedef enum
{
    /* Block continuously waiting for events */
    CDBUS_RUN_WAIT,
    /* Process outstanding events but don't block for new ones */
    CDBUS_RUN_NO_WAIT,
    /* Block waiting for events then return */
    CDBUS_RUN_ONCE
} cdbus_RunOption;


typedef void (*cdbus_WakeupFunc)(CDBUS_DISPATCHER_P, void*);


CDBUS_EXPORT cdbus_Dispatcher* cdbus_dispatcherNew(EV_P_
                                        cdbus_Bool ownsLoop,
                                        cdbus_WakeupFunc wakeupFunc,
                                        void* wakeupData);
CDBUS_EXPORT void cdbus_dispatcherUnref(CDBUS_DISPATCHER_P);
CDBUS_EXPORT cdbus_Dispatcher* cdbus_dispatcherRef(CDBUS_DISPATCHER_P);
CDBUS_EXPORT cdbus_HResult cdbus_dispatcherRun(CDBUS_DISPATCHER_P,
                                            cdbus_RunOption runOpt);
CDBUS_EXPORT cdbus_HResult cdbus_dispatcherRunWithData(CDBUS_DISPATCHER_P,
                                            cdbus_RunOption runOpt,
                                            void* dispData);
CDBUS_EXPORT cdbus_HResult cdbus_dispatcherStop(CDBUS_DISPATCHER_P);
CDBUS_EXPORT void cdbus_dispatcherBreak(CDBUS_DISPATCHER_P);

/* Only to be called by client code after the wake up function has been called on the client. */
CDBUS_EXPORT void cdbus_dispatcherInvokePending(CDBUS_DISPATCHER_P);

CDBUS_EXPORT cdbus_HResult cdbus_dispatcherAddWatch(CDBUS_DISPATCHER_P,
                       struct cdbus_Watch* watch);
CDBUS_EXPORT cdbus_HResult cdbus_dispatcherRemoveWatch(CDBUS_DISPATCHER_P,
                       struct cdbus_Watch* watch);
CDBUS_EXPORT cdbus_HResult cdbus_dispatcherAddTimeout(CDBUS_DISPATCHER_P,
                       struct cdbus_Timeout* timeout);
CDBUS_EXPORT cdbus_HResult cdbus_dispatcherRemoveTimeout(CDBUS_DISPATCHER_P,
                       struct cdbus_Timeout* timeout);

CDBUS_END_DECLS


#endif /* Guard for CDBUS_DISPATCHER_H_ */
