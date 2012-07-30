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
 * @file           watch.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of a D-Bus watch class.
 *******************************************************************************
 */

#ifndef CDBUS_WATCH_H_
#define CDBUS_WATCH_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct cdbus_Dispatcher;
typedef struct cdbus_Watch cdbus_Watch;
typedef cdbus_Bool (*cdbus_WatchHandler)(cdbus_Watch* w, cdbus_UInt32, void*);

CDBUS_EXPORT cdbus_Watch* cdbus_watchNew(struct cdbus_Dispatcher* dispatcher,
                                     cdbus_Descriptor fd, cdbus_UInt32 flags,
                                     cdbus_WatchHandler h, void* data);
CDBUS_EXPORT cdbus_Watch* cdbus_watchRef(cdbus_Watch* w);
CDBUS_EXPORT void cdbus_watchUnref(cdbus_Watch* w);
CDBUS_EXPORT cdbus_UInt32 cdbus_watchGetFlags(cdbus_Watch* w);
CDBUS_EXPORT cdbus_HResult cdbus_watchSetFlags(cdbus_Watch* w, cdbus_UInt32 flags);
CDBUS_EXPORT cdbus_Bool cdbus_watchIsEnabled(cdbus_Watch* w);
CDBUS_EXPORT cdbus_HResult cdbus_watchEnable(cdbus_Watch* w, cdbus_Bool option);
CDBUS_EXPORT void* cdbus_watchGetData(cdbus_Watch* w);
CDBUS_EXPORT void cdbus_watchSetData(cdbus_Watch* w, void* data);


CDBUS_END_DECLS

#endif /* Guard for CDBUS_WATCH_H_ */
