/*===========================================================================
 * 
 * Project         cdbus
 *
 * Released under the MIT License (MIT)
 * Copyright (c) 2013 XS-Embedded LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
 * NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *===========================================================================
 *===========================================================================
 * @file           object.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of the D-Bus object representation.
 *===========================================================================
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
