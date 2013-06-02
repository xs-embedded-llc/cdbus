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
 * @file           interface-priv.h        
 * @author         Glenn Schmottlach
 * @brief          Brief description
 *===========================================================================
 */

#ifndef CDBUS_INTERFACE_PRIV_H_
#define CDBUS_INTERFACE_PRIV_H_

#include "cdbus/interface.h"
#include "cdbus/object.h"
#include "cdbus/connection.h"
#include "cdbus/atomic-ops.h"
#include "queue.h"
#include "mutex.h"

CDBUS_BEGIN_DECLS

typedef struct cdbus_InterfaceArgs
{
    LIST_ENTRY(cdbus_InterfaceArgs) link;
    cdbus_Char*                     name;
    cdbus_Char*                     signature;
    cdbus_TransferDir               xferDir;
} cdbus_InterfaceArgs;


typedef struct cdbus_InterfaceItem
{
    LIST_ENTRY(cdbus_InterfaceItem) link;
    cdbus_Char*                     name;
    cdbus_InterfaceArgs*            args;
    cdbus_UInt32                    nArgs;

} cdbus_InterfaceItem;

typedef cdbus_InterfaceItem cdbus_InterfaceMethod;
typedef cdbus_InterfaceItem cdbus_InterfaceSignal;

typedef struct cdbus_InterfaceProperty
{
    LIST_ENTRY(cdbus_InterfaceProperty) link;
    cdbus_Char*                         name;
    cdbus_Char*                         signature;
    cdbus_Bool                          read;
    cdbus_Bool                          write;
} cdbus_InterfaceProperty;


struct cdbus_Interface
{
    cdbus_Atomic                    refCnt;
    cdbus_InterfaceMessageHandler   handler;
    void*                           userData;
    cdbus_Char*                     name;
    CDBUS_LOCK_DECLARE(lock);
    LIST_HEAD(cdbus_InfItemHead, cdbus_InterfaceItem)       methods;
    struct cdbus_InfItemHead                                signals;
    LIST_HEAD(cdbus_InfPropHead, cdbus_InterfaceProperty)   props;
};


DBusHandlerResult cdbus_interfaceHandleMessage(cdbus_Interface* intf,
                            cdbus_Object* obj, cdbus_Connection* conn,
                            DBusMessage* msg);

CDBUS_END_DECLS


#endif /* Guard for CDBUS_INTERFACE_PRIV_H_ */
