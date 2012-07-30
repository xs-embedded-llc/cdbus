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
 * @file           timeout-priv.h
 * @author         Glenn Schmottlach
 * @brief          Private declaration of timeout class.
 *******************************************************************************
 */

#ifndef CDBUS_TIMEOUT_PRIV_H_
#define CDBUS_TIMEOUT_PRIV_H_

#include "cdbus/types.h"
#include "cdbus/timeout.h"
#include "queue.h"
#include "mutex.h"
#include "ev.h"


struct cdbus_Timeout
{
    cdbus_Dispatcher*           dispatcher;
    cdbus_Atomic                refCnt;
    cdbus_Mutex*                lock;
    ev_timer                    timerWatcher;
    cdbus_Bool                  repeat;
    cdbus_TimeoutHandler        handler;
    void*                       data;
    LIST_ENTRY(cdbus_Timeout)   link;
};


#endif /* Guard for CDBUS_TIMEOUT_PRIV_H_ */
