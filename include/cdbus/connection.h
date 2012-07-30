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
 * @file           connection.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of D-Bus connection wrapper class.
 *******************************************************************************
 */

#ifndef CDBUS_CONNECTION_H_
#define CDBUS_CONNECTION_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"
#include "cdbus/object.h"
#include "dbus/dbus.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
typedef struct cdbus_Connection cdbus_Connection;
struct cdbus_Dispatcher;

CDBUS_EXPORT cdbus_Connection* cdbus_connectionNew(struct cdbus_Dispatcher* disp);
CDBUS_EXPORT void cdbus_connectionUnref(cdbus_Connection* conn);
CDBUS_EXPORT cdbus_Connection* cdbus_connectionRef(cdbus_Connection* conn);
CDBUS_EXPORT cdbus_HResult cdbus_connectionOpen(cdbus_Connection* conn, const cdbus_Char* address,
                                                cdbus_Bool private, cdbus_Bool exitOnDisconnect);
CDBUS_EXPORT cdbus_HResult cdbus_connectionOpenStandard(cdbus_Connection* conn, DBusBusType busType,
                                                cdbus_Bool private, cdbus_Bool exitOnDisconnect);
CDBUS_EXPORT cdbus_HResult cdbus_connectionClose(cdbus_Connection* conn);
CDBUS_EXPORT DBusConnection* cdbus_connectionGetDBus(cdbus_Connection* conn);
CDBUS_EXPORT cdbus_Bool cdbus_connectionRegisterObject(cdbus_Connection* conn, cdbus_Object* obj);
CDBUS_EXPORT cdbus_Bool cdbus_connectionUnregisterObject(cdbus_Connection* conn, const cdbus_Char* path);



CDBUS_END_DECLS

#endif /* Guard for CDBUS_CONNECTION_H_ */
