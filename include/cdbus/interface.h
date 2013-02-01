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
 * @file           interface.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of the D-Bus interface class.
 *******************************************************************************
 */

#ifndef CDBUS_INTERFACE_H_
#define CDBUS_INTERFACE_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"
#include "cdbus/stringbuffer.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct cdbus_Object;
struct cdbus_Connection;

typedef struct cdbus_Interface cdbus_Interface;

typedef enum
{
    CDBUS_XFER_IN,
    CDBUS_XFER_OUT
} cdbus_TransferDir;


typedef struct cdbus_DbusIntrospectArgs
{
    cdbus_Char*         name;
    cdbus_Char*         signature;
    cdbus_TransferDir   xferDir;
} cdbus_DbusIntrospectArgs;


typedef struct cdbus_DbusIntrospectItem
{
    cdbus_Char*                 name;
    cdbus_DbusIntrospectArgs*   args;
    cdbus_UInt32                nArgs;

} cdbus_DbusIntrospectItem;


typedef struct cdbus_DbusIntrospectProperty
{
    cdbus_Char* name;
    cdbus_Char* signature;
    cdbus_Bool  read;
    cdbus_Bool  write;
} cdbus_DbusIntrospectProperty;


typedef cdbus_DbusIntrospectItem cdbus_DbusIntrospectMethod;
typedef cdbus_DbusIntrospectItem cdbus_DbusIntrospectSignal;
typedef DBusHandlerResult (*cdbus_InterfaceMessageHandler)(struct cdbus_Connection*,
                        struct cdbus_Object*, DBusMessage*, void*);

CDBUS_EXPORT cdbus_Interface* cdbus_interfaceNew(const cdbus_Char* name,
                        cdbus_InterfaceMessageHandler handler,
                        void* userData);
CDBUS_EXPORT cdbus_Interface* cdbus_interfaceRef(cdbus_Interface* intf);
CDBUS_EXPORT void cdbus_interfaceUnref(cdbus_Interface* intf);
CDBUS_EXPORT const cdbus_Char* cdbus_interfaceGetName(cdbus_Interface* intf);
CDBUS_EXPORT void cdbus_interfaceSetData(cdbus_Interface* intf, void* data);
CDBUS_EXPORT void* cdbus_interfaceGetData(cdbus_Interface* intf);
CDBUS_EXPORT cdbus_Bool cdbus_interfaceRegisterMethods(cdbus_Interface* intf,
                            const cdbus_DbusIntrospectItem* methods,
                            cdbus_UInt32 numMethods);
CDBUS_EXPORT cdbus_Bool cdbus_interfaceClearMethods(cdbus_Interface* intf);
CDBUS_EXPORT cdbus_Bool cdbus_interfaceRegisterSignals(cdbus_Interface* intf,
                            const cdbus_DbusIntrospectItem* signals,
                            cdbus_UInt32 numSignals);
CDBUS_EXPORT cdbus_Bool cdbus_interfaceClearSignals(cdbus_Interface* intf);
CDBUS_EXPORT cdbus_Bool cdbus_interfaceRegisterProperties(cdbus_Interface* intf,
                            const cdbus_DbusIntrospectProperty* properties,
                            cdbus_UInt32 numProperties);
CDBUS_EXPORT cdbus_Bool cdbus_interfaceClearProperties(cdbus_Interface* intf);
CDBUS_EXPORT cdbus_StringBuffer* cdbus_interfaceIntrospect(cdbus_Interface* intf);


CDBUS_END_DECLS


#endif /* Guard for CDBUS_INTERFACE_H_ */
