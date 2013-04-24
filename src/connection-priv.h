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
 * @file           connection-priv.h        
 * @author         Glenn Schmottlach
 * @brief          Private declaration of connection class.
 *===========================================================================
 */

#ifndef CDBUS_CONNECTION_PRIV_H_
#define CDBUS_CONNECTION_PRIV_H_

#include "cdbus/cdbus.h"
#include "queue.h"
#include "mutex.h"
#include "match.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct cdbus_Connection
{
    cdbus_Dispatcher*               dispatcher;
    DBusConnection*                 dbusConn;
    cdbus_Bool                      isPrivate;
    cdbus_Match*                    nextMatch;
    cdbus_Atomic                    refCnt;
    LIST_HEAD(cdbus_MatchHead,
              cdbus_Match)          matches;
    CDBUS_LOCK_DECLARE(lock);
    LIST_ENTRY(cdbus_Connection)    link;
};


cdbus_Connection* cdbus_connectionNew(struct cdbus_Dispatcher* disp,
                                    DBusConnection* dbusConn,
                                    cdbus_Bool isPrivate);

DBusHandlerResult cdbus_connectionFilterHandler(DBusConnection* dbusConn,
                         DBusMessage* msg, void* data);

DBusHandlerResult cdbus_connectionDispatchMatches(cdbus_Connection* conn,
                                                         DBusMessage* msg);

CDBUS_END_DECLS

#endif /* Guard for CDBUS_CONNECTION_PRIV_H_ */
