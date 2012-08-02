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
 * @file           dispatcher.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of D-Bus dispatcher class.
 *******************************************************************************
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


typedef enum
{
    /* Breaks all nested calls to the run loop */
    CDBUS_BREAK_ALL,
    /* Breaks only innermost nested run loop */
    CDBUS_BREAK_ONE
} cdbus_BreakOption;


typedef void (*cdbus_WakeupFunc)(CDBUS_DISPATCHER_P, void*);


CDBUS_EXPORT cdbus_Dispatcher* cdbus_dispatcherNew(EV_P_
                                        cdbus_Bool ownsLoop,
                                        cdbus_WakeupFunc wakeupFunc,
                                        void* wakeupData);
CDBUS_EXPORT void cdbus_dispatcherUnref(CDBUS_DISPATCHER_P);
CDBUS_EXPORT cdbus_Dispatcher* cdbus_dispatcherRef(CDBUS_DISPATCHER_P);
CDBUS_EXPORT cdbus_HResult cdbus_dispatcherRun(CDBUS_DISPATCHER_P,
                                            cdbus_RunOption runOpt);
CDBUS_EXPORT cdbus_HResult cdbus_dispatcherBreak(CDBUS_DISPATCHER_P, cdbus_BreakOption opt);

/* Only to be called from a secondary thread after the wakeup function has been called. */
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
