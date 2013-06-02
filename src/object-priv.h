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
 * @file           object-priv.h        
 * @author         Glenn Schmottlach
 * @brief          Private definitions of the object class.
 *===========================================================================
 */

#include "cdbus/connection.h"
#include "cdbus/object.h"
#include "cdbus/atomic-ops.h"
#include "dbus/dbus.h"
#include "string-pointer-map.h"
#include "mutex.h"
#include "interface-priv.h"

CDBUS_BEGIN_DECLS

struct cdbus_Object
{
    cdbus_Char*                 objPath;
    void*                       userData;
    cdbus_Atomic                refCnt;
    CDBUS_LOCK_DECLARE(lock);
    cdbus_StrPtrMap*            interfaces;
    cdbus_ObjectMessageHandler  handler;
};

DBusHandlerResult cdbus_objectMessageDispatcher(cdbus_Object* obj,
                            cdbus_Connection* conn, DBusMessage* msg);

CDBUS_END_DECLS
