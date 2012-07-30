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


CDBUS_EXPORT cdbus_Dispatcher* cdbus_dispatcherNew(EV_P_ cdbus_Bool ownsLoop,
                                  cdbus_WakeupFunc wakeupFunc, void* wakeupData);
CDBUS_EXPORT void cdbus_dispatcherUnref(CDBUS_DISPATCHER_P);
CDBUS_EXPORT cdbus_Dispatcher* cdbus_dispatcherRef(CDBUS_DISPATCHER_P);

CDBUS_END_DECLS


#endif /* Guard for CDBUS_DISPATCHER_H_ */
