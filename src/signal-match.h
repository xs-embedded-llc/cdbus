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
 * @file           signal-match.h        
 * @author         Glenn Schmottlach
 * @brief          Define an object representing a signal match and the
 *                 operations on it.
 *******************************************************************************
 */

#ifndef CDBUS_SIGNAL_MATCH_H_
#define CDBUS_SIGNAL_MATCH_H_

#include "cdbus/cdbus.h"
#include "dbus/dbus.h"
#include "queue.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct cdbus_Connection;


typedef struct cdbus_SignalMatch
{
    cdbus_connectionSignalHandler   handler;
    void*                           userData;
    cdbus_SignalRule                rule;
    cdbus_Char*                     ruleStr;
    cdbus_UInt16                    nFilterArgs;
    cdbus_Atomic                    refCnt;
    LIST_ENTRY(cdbus_SignalMatch)   link;
} cdbus_SignalMatch;

cdbus_SignalMatch* cdbus_signalMatchNew(cdbus_connectionSignalHandler handler,
                                        void* userData,
                                        const cdbus_SignalRule* rule);
cdbus_SignalMatch* cdbus_signalMatchRef(cdbus_SignalMatch* sigMatch);
void cdbus_signalMatchUnref(cdbus_SignalMatch* sigMatch);
cdbus_Bool cdbus_signalMatchIsMatch(cdbus_SignalMatch* sigMatch, DBusMessage* msg);
void cdbus_signalMatchDispatch(struct cdbus_Connection* conn,
                                cdbus_SignalMatch* sigMatch,
                                DBusMessage* msg);
const cdbus_Char* cdbus_signalMatchGetRule(cdbus_SignalMatch* sigMatch);
cdbus_Bool cdbus_signalMatchAddFilter(cdbus_SignalMatch* sigMatch, struct cdbus_Connection* conn);
cdbus_Bool cdbus_signalMatchRemoveFilter(cdbus_SignalMatch* sigMatch, struct cdbus_Connection* conn);

CDBUS_END_DECLS

#endif /* Guard for CDBUS_SIGNAL_MATCH_H_ */
