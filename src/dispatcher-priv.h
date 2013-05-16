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
 * @file           dispatcher-priv.h        
 * @author         Glenn Schmottlach
 * @brief          Private dispatcher declarations.
 *===========================================================================
 */

#ifndef CDBUS_DISPATCHER_PRIV_H_
#define CDBUS_DISPATCHER_PRIV_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"
#include "queue.h"
#include "connection-priv.h"
#include "watch-priv.h"
#include "timeout-priv.h"
#include "mutex.h"
#include "semaphore.h"

CDBUS_BEGIN_DECLS


struct cdbus_Dispatcher
{
#if EV_MULTIPLICITY
    EV_P;
#endif
    cdbus_Bool                                  ownsLoop;
    LIST_HEAD(cdbus_ConnHead,cdbus_Connection)  connections;
    LIST_HEAD(cdbus_WatchHead, cdbus_Watch)     watches;
    LIST_HEAD(cdbus_TimeoutHead, cdbus_Timeout) timeouts;
    cdbus_Atomic                                refCnt;
    ev_async                                    asyncWatch;
    CDBUS_LOCK_DECLARE(lock);
    cdbus_WakeupFunc                            wakeupFunc;
    void*                                       wakeupData;
    cdbus_FinalizerFunc                         finalizerFunc;
    void*                                       finalizerData;
    cdbus_Semaphore*                            barrier;
    cdbus_Bool                                  dispatchNeeded;
    volatile cdbus_Bool                         exitLoop;
};

cdbus_HResult cdbus_dispatcherAddConnection(CDBUS_DISPATCHER_P,
                       struct cdbus_Connection* conn);
cdbus_HResult cdbus_dispatcherRemoveConnection(CDBUS_DISPATCHER_P,
                       struct cdbus_Connection* conn);
struct cdbus_Connection* cdbus_dispatcherGetDbusConnOwner(CDBUS_DISPATCHER_P,
                                                DBusConnection* dbusConn);
void cdbus_dispatcherWakeup(CDBUS_DISPATCHER_P);

cdbus_HResult cdbus_dispatcherAddWatch(CDBUS_DISPATCHER_P,
                       struct cdbus_Watch* watch);
cdbus_HResult cdbus_dispatcherRemoveWatch(CDBUS_DISPATCHER_P,
                       struct cdbus_Watch* watch);
cdbus_HResult cdbus_dispatcherAddTimeout(CDBUS_DISPATCHER_P,
                       struct cdbus_Timeout* timeout);
cdbus_HResult cdbus_dispatcherRemoveTimeout(CDBUS_DISPATCHER_P,
                       struct cdbus_Timeout* timeout);

CDBUS_END_DECLS

#endif /* Guard for CDBUS_DISPATCHER_PRIV_H_ */
