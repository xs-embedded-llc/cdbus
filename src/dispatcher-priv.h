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
 * @file           dispatcher-priv.h        
 * @author         Glenn Schmottlach
 * @brief          Private dispatcher declarations.
 *******************************************************************************
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
#include "condvar.h"

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
    cdbus_Mutex*                                lock;
    cdbus_CondVar*                              cv;
    cdbus_WakeupFunc                            wakeupFunc;
    void*                                       wakeupData;
    cdbus_Bool                                  dispatchNeeded;
};

cdbus_HResult cdbus_dispatcherAddConnection(CDBUS_DISPATCHER_P,
                       struct cdbus_Connection* conn);
cdbus_HResult cdbus_dispatcherRemoveConnection(CDBUS_DISPATCHER_P,
                       struct cdbus_Connection* conn);
cdbus_HResult cdbus_dispatcherAddWatch(CDBUS_DISPATCHER_P,
                       struct cdbus_Watch* watch);
cdbus_HResult cdbus_dispatcherRemoveWatch(CDBUS_DISPATCHER_P,
                       struct cdbus_Watch* watch);
cdbus_HResult cdbus_dispatcherAddTimeout(CDBUS_DISPATCHER_P,
                       struct cdbus_Timeout* timeout);
cdbus_HResult cdbus_dispatcherRemoveTimeout(CDBUS_DISPATCHER_P,
                       struct cdbus_Timeout* timeout);
void cdbus_dispatcherWakeup(cdbus_Dispatcher* d);


CDBUS_END_DECLS

#endif /* Guard for CDBUS_DISPATCHER_PRIV_H_ */