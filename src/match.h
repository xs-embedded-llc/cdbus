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
 * @file           match.h
 * @author         Glenn Schmottlach
 * @brief          Define an object representing a match rule and the
 *                 operations on it.
 *******************************************************************************
 */

#ifndef CDBUS_MATCH_H_
#define CDBUS_MATCH_H_

#include "cdbus/cdbus.h"
#include "dbus/dbus.h"
#include "queue.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct cdbus_Connection;


typedef struct cdbus_Match
{
    cdbus_connectionMatchHandler    handler;
    void*                           userData;
    cdbus_MatchRule                 rule;
    cdbus_Char*                     ruleStr;
    cdbus_UInt16                    nFilterArgs;
    cdbus_Atomic                    refCnt;
    LIST_ENTRY(cdbus_Match)         link;
} cdbus_Match;

cdbus_Match* cdbus_matchNew(cdbus_connectionMatchHandler handler,
                                        void* userData,
                                        const cdbus_MatchRule* rule);
cdbus_Match* cdbus_matchRef(cdbus_Match* match);
void cdbus_matchUnref(cdbus_Match* match);
cdbus_Bool cdbus_matchIsMatch(cdbus_Match* match, DBusMessage* msg);
void cdbus_matchDispatch(struct cdbus_Connection* conn,
                                cdbus_Match* match,
                                DBusMessage* msg);
const cdbus_Char* cdbus_matchGetRule(cdbus_Match* match);
cdbus_Bool cdbus_matchAddFilter(cdbus_Match* match, struct cdbus_Connection* conn);
cdbus_Bool cdbus_matchRemoveFilter(cdbus_Match* match, struct cdbus_Connection* conn);

CDBUS_END_DECLS

#endif /* Guard for CDBUS_MATCH_H_ */