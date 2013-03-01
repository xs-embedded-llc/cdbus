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
