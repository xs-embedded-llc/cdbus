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
 * @file           object-priv.h        
 * @author         Glenn Schmottlach
 * @brief          Private definitions of the object class.
 *******************************************************************************
 */

#include "cdbus/connection.h"
#include "cdbus/object.h"
#include "dbus/dbus.h"
#include "string-pointer-map.h"
#include "atomic-ops.h"
#include "mutex.h"
#include "interface-priv.h"

CDBUS_BEGIN_DECLS

struct cdbus_Object
{
    cdbus_Char*                 objPath;
    void*                       userData;
    cdbus_Atomic                refCnt;
    cdbus_Mutex*                lock;
    cdbus_StrPtrMap*            interfaces;
    cdbus_ObjectMessageHandler  handler;
};

DBusHandlerResult cdbus_objectMessageDispatcher(cdbus_Object* obj,
                            cdbus_Connection* conn, DBusMessage* msg);

CDBUS_END_DECLS
