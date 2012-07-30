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
 * @file           interface-priv.h        
 * @author         Glenn Schmottlach
 * @brief          Brief description
 *******************************************************************************
 */

#ifndef CDBUS_INTERFACE_PRIV_H_
#define CDBUS_INTERFACE_PRIV_H_

#include "cdbus/interface.h"
#include "cdbus/object.h"
#include "cdbus/connection.h"
#include "queue.h"
#include "atomic-ops.h"
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
    LIST_ENTRY(cdbus_Interface)     link;
    cdbus_Atomic                    refCnt;
    cdbus_Mutex*                    lock;
    cdbus_InterfaceMessageHandler   handler;
    void*                           userData;
    cdbus_Char*                     name;
    LIST_HEAD(cdbus_InfItemHead, cdbus_InterfaceItem)       methods;
    struct cdbus_InfItemHead                                signals;
    LIST_HEAD(cdbus_InfPropHead, cdbus_InterfaceProperty)   props;
};


DBusHandlerResult cdbus_interfaceHandleMessage(cdbus_Interface* intf,
                            cdbus_Object* obj, cdbus_Connection* conn,
                            DBusMessage* msg);

CDBUS_END_DECLS


#endif /* Guard for CDBUS_INTERFACE_PRIV_H_ */
