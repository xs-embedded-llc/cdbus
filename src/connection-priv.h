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
 * @file           connection-priv.h        
 * @author         Glenn Schmottlach
 * @brief          Private declaration of connection class.
 *******************************************************************************
 */

#ifndef CDBUS_CONNECTION_PRIV_H_
#define CDBUS_CONNECTION_PRIV_H_

#include "cdbus/cdbus.h"
#include "queue.h"
#include "mutex.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct cdbus_Connection
{
    cdbus_Dispatcher*               dispatcher;
    DBusConnection*                 dbusConn;
    cdbus_Bool                      isPrivate;
    cdbus_Atomic                    refCnt;
    cdbus_Mutex*                    lock;
    LIST_ENTRY(cdbus_Connection)    link;
};


cdbus_Connection* cdbus_connectionNew(struct cdbus_Dispatcher* disp,
                                    DBusConnection* dbusConn,
                                    cdbus_Bool isPrivate);

CDBUS_END_DECLS

#endif /* Guard for CDBUS_CONNECTION_PRIV_H_ */
