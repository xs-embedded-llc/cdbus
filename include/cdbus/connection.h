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
 * @file           connection.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of D-Bus connection wrapper class.
 *===========================================================================
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

typedef enum {
    CDBUS_FILTER_ARG_INVALID = 0,
    CDBUS_FILTER_ARG = 1,
    CDBUS_FILTER_ARG_PATH = 2,
} cdbus_FilterArgType;

typedef struct cdbus_FilterArgItem
{
    cdbus_FilterArgType argType;
    cdbus_UInt8         argN;
    cdbus_Char*         value;
} cdbus_FilterArgItem;

typedef enum {
    CDBUS_MATCH_MSG_ANY,
    CDBUS_MATCH_MSG_METHOD_CALL,
    CDBUS_MATCH_MSG_METHOD_RETURN,
    CDBUS_MATCH_MSG_SIGNAL,
    CDBUS_MATCH_MSG_ERROR
} cdbus_MatchMsgType;

typedef struct cdbus_MatchRule
{
    cdbus_MatchMsgType      msgType;
    cdbus_Char*             member;
    cdbus_Char*             objInterface;
    cdbus_Char*             sender;
    cdbus_Char*             path;
    cdbus_Bool              treatPathAsNamespace;
    cdbus_Char*             arg0Namespace;
    cdbus_FilterArgItem*    filterArgs;
    cdbus_Bool              eavesdrop;
} cdbus_MatchRule;

typedef void (*cdbus_connectionMatchHandler)(cdbus_Connection* conn, cdbus_Handle hnd,
                                                DBusMessage* msg, void* userData);

CDBUS_EXPORT cdbus_Connection* cdbus_connectionOpen(struct cdbus_Dispatcher* disp, const cdbus_Char* address,
                                                cdbus_Bool private, cdbus_Bool exitOnDisconnect);
CDBUS_EXPORT cdbus_Connection* cdbus_connectionOpenStandard(struct cdbus_Dispatcher* disp, DBusBusType busType,
                                                cdbus_Bool private, cdbus_Bool exitOnDisconnect);
CDBUS_EXPORT cdbus_HResult cdbus_connectionClose(cdbus_Connection* conn);
CDBUS_EXPORT cdbus_Connection* cdbus_connectionRef(cdbus_Connection* conn);
CDBUS_EXPORT void cdbus_connectionUnref(cdbus_Connection* conn);
CDBUS_EXPORT DBusConnection* cdbus_connectionGetDBus(cdbus_Connection* conn);
CDBUS_EXPORT cdbus_Bool cdbus_connectionGetDescriptor(cdbus_Connection* conn, cdbus_Descriptor* descr);
CDBUS_EXPORT cdbus_Bool cdbus_connectionRegisterObject(cdbus_Connection* conn, cdbus_Object* obj);
CDBUS_EXPORT cdbus_Bool cdbus_connectionUnregisterObject(cdbus_Connection* conn, const cdbus_Char* path);
CDBUS_EXPORT cdbus_Bool cdbus_connectionSendWithReply(cdbus_Connection* conn, DBusMessage* msg,
                                                    DBusPendingCall** pending, cdbus_Int32 timeout,
                                                    DBusPendingCallNotifyFunction  notifyFunc,
                                                    void* userData, DBusFreeFunction freeUserDataFunc);

CDBUS_EXPORT cdbus_Handle cdbus_connectionRegMatchHandler(
                                                    cdbus_Connection* conn,
                                                    cdbus_connectionMatchHandler handler,
                                                    void* userData,
                                                    const cdbus_MatchRule* rule,
                                                    cdbus_HResult* hResult);
CDBUS_EXPORT cdbus_HResult cdbus_connectionUnregMatchHandler(
                                                    cdbus_Connection* conn,
                                                    cdbus_Handle regHnd);

CDBUS_EXPORT cdbus_Bool cdbus_connectionLock(cdbus_Connection* conn);
CDBUS_EXPORT cdbus_Bool cdbus_connectionUnlock(cdbus_Connection* conn);



CDBUS_END_DECLS

#endif /* Guard for CDBUS_CONNECTION_H_ */
