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
 * @file           dbus-watch-ctrl.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of D-Bus related watch control functions.
 *******************************************************************************
 */

#ifndef CDBUS_DBUS_WATCH_CTRL_H_
#define CDBUS_DBUS_WATCH_CTRL_H_

#include "dbus/dbus.h"
#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

dbus_bool_t cdbus_watchAddHandler(DBusWatch* dbusWatch, void* data);
void cdbus_watchRemoveHandler(DBusWatch* dbusWatch, void* data);
void cdbus_watchToggleHandler(DBusWatch* dbusWatch, void* data);

CDBUS_END_DECLS

#endif /* Guard for CDBUS_DBUS_WATCH_CTRL_H_ */
