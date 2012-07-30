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
 * @file           timeout.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of a D-Bus timeout class.
 *******************************************************************************
 */

#ifndef CDBUS_TIMEOUT_H_
#define CDBUS_TIMEOUT_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"
#include "dbus/dbus.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct cdbus_Dispatcher;
typedef struct cdbus_Timeout cdbus_Timeout;
typedef cdbus_Bool (*cdbus_TimeoutHandler)(cdbus_Timeout* t, void*);

CDBUS_EXPORT cdbus_Timeout* cdbus_timeoutNew(struct cdbus_Dispatcher* dispatcher,
                                cdbus_Int32 msecInterval, cdbus_Bool repeat,
                                cdbus_TimeoutHandler h, void* data);
CDBUS_EXPORT cdbus_Timeout* cdbus_timeoutRef(cdbus_Timeout* t);
CDBUS_EXPORT void cdbus_timeoutUnref(cdbus_Timeout* t);

CDBUS_EXPORT cdbus_Bool cdbus_timeoutIsEnabled(cdbus_Timeout* t);
CDBUS_EXPORT cdbus_HResult cdbus_timeoutEnable(cdbus_Timeout* t, cdbus_Bool option);
CDBUS_EXPORT cdbus_Int32 cdbus_timeoutInterval(cdbus_Timeout* t);
CDBUS_EXPORT cdbus_HResult cdbus_timeoutSetInterval(cdbus_Timeout* t,
                                                cdbus_Int32 msecInterval);
CDBUS_EXPORT cdbus_Bool cdbus_timeoutGetRepeat(cdbus_Timeout* t);
CDBUS_EXPORT void cdbus_timeoutSetRepeat(cdbus_Timeout* t, cdbus_Bool repeat);
CDBUS_EXPORT void* cdbus_timeoutGetData(cdbus_Timeout* t);
CDBUS_EXPORT void cdbus_timeoutSetData(cdbus_Timeout* t, void* data);



CDBUS_END_DECLS

#endif /* Guard for CDBUS_TIMEOUT_H_ */
