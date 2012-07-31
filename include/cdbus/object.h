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
 * @file           object.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of the D-Bus object representation.
 *******************************************************************************
 */

#ifndef CDBUS_OBJECT_H_
#define CDBUS_OBJECT_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"
#include "cdbus/stringbuffer.h"
#include "dbus/dbus.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct cdbus_Interface;
struct cdbus_Connection;

typedef struct cdbus_Object cdbus_Object;
typedef DBusHandlerResult (*cdbus_ObjectMessageHandler)(struct cdbus_Object*,
                            struct cdbus_Connection*, DBusMessage*);

CDBUS_EXPORT cdbus_Object* cdbus_objectNew(const cdbus_Char* objPath,
                        cdbus_ObjectMessageHandler defaultHandler,
                        void* userData);
CDBUS_EXPORT cdbus_Object* cdbus_objectRef(cdbus_Object* obj);
CDBUS_EXPORT void cdbus_objectUnref(cdbus_Object* obj);

CDBUS_EXPORT cdbus_HResult cdbus_objectCopyPath(cdbus_Object* obj, cdbus_Char* buf, cdbus_UInt32* size);
CDBUS_EXPORT const cdbus_Char* cdbus_objectGetPath(cdbus_Object* obj);
CDBUS_EXPORT void cdbus_objectSetData(cdbus_Object* obj, void* data);
CDBUS_EXPORT void* cdbus_objectGetData(cdbus_Object* obj);
CDBUS_EXPORT cdbus_Bool cdbus_objectAddInterface(cdbus_Object* obj, struct cdbus_Interface* intf);
CDBUS_EXPORT cdbus_Bool cdbus_objectRemoveInterface(cdbus_Object* obj, const cdbus_Char* name);
CDBUS_EXPORT cdbus_StringBuffer* cdbus_objectIntrospect(cdbus_Object* obj,
                                        struct cdbus_Connection* conn, const cdbus_Char* path);




CDBUS_END_DECLS


#endif /* Guard for CDBUS_OBJECT_H_ */
