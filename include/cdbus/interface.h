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
 * @file           interface.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of the D-Bus interface class.
 *===========================================================================
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
