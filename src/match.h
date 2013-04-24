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
 * @file           match.h
 * @author         Glenn Schmottlach
 * @brief          Define an object representing a match rule and the
 *                 operations on it.
 *===========================================================================
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
