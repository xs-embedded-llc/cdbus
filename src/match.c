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
 * @file           match.c
 * @author         Glenn Schmottlach
 * @brief          Implementation of an object representing a match rule and
 *                 the operations on it.
 *******************************************************************************
 */
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include "match.h"
#include "trace.h"
#include "alloc.h"
#include "atomic-ops.h"

static
int compareArgs
    (
    const void* a,
    const void* b
    )
{
    cdbus_FilterArgItem* itemA = (cdbus_FilterArgItem*)a;
    cdbus_FilterArgItem* itemB = (cdbus_FilterArgItem*)b;

    return (int)itemA->argN - (int)itemB->argN;
}


cdbus_Match*
cdbus_matchNew
    (
    cdbus_connectionMatchHandler    handler,
    void*                           userData,
    const cdbus_MatchRule*          rule
    )
{
    cdbus_Match* obj = NULL;
    cdbus_Int32 idx;
    cdbus_Bool allocError = CDBUS_FALSE;
    cdbus_FilterArgItem* item;

    if ( (NULL != handler) && (NULL != rule) )
    {
        obj = cdbus_calloc(1, sizeof(*obj));
        if ( NULL != obj )
        {
            obj->handler = handler;
            obj->userData = userData;
            obj->rule.msgType = rule->msgType;
            obj->rule.member = cdbus_strDup(rule->member);
            obj->rule.sender = cdbus_strDup(rule->sender);
            obj->rule.objInterface = cdbus_strDup(rule->objInterface);
            obj->rule.path = cdbus_strDup(rule->path);
            obj->rule.arg0Namespace = cdbus_strDup(rule->arg0Namespace);
            obj->rule.treatPathAsNamespace = rule->treatPathAsNamespace;
            obj->rule.localObjPath = cdbus_strDup(rule->localObjPath);
            obj->rule.eavesdrop = rule->eavesdrop;

            /* Count the number of filter arguments */
            obj->nFilterArgs = 0U;
            item = rule->filterArgs;
            while ( (NULL != item) &&
                (CDBUS_FILTER_ARG_INVALID != item->argType) &&
                (obj->nFilterArgs < DBUS_MAXIMUM_MATCH_RULE_ARG_NUMBER) )
            {
                obj->nFilterArgs++;
                ++item;
            }

            if ( 0 < obj->nFilterArgs )
            {
                obj->rule.filterArgs = cdbus_calloc(obj->nFilterArgs, sizeof(*(obj->rule.filterArgs)));
                if ( NULL == obj->rule.filterArgs )
                {
                    allocError = CDBUS_TRUE;
                    obj->nFilterArgs = 0U;
                }
                else
                {
                    for ( idx = 0; idx < obj->nFilterArgs; ++idx )
                    {
                        obj->rule.filterArgs[idx].argType = rule->filterArgs[idx].argType;
                        obj->rule.filterArgs[idx].argN = rule->filterArgs[idx].argN;
                        obj->rule.filterArgs[idx].value = cdbus_strDup(rule->filterArgs[idx].value);
                    }

                    /* Sort these in order of increasing argument index */
                    qsort(obj->rule.filterArgs,
                        obj->nFilterArgs, sizeof(*obj->rule.filterArgs), compareArgs);
                }
            }

            obj = cdbus_matchRef(obj);
            if ( allocError )
            {
                cdbus_matchUnref(obj);
                obj = NULL;
            }
        }
    }
    return obj;
}


cdbus_Match*
cdbus_matchRef
    (
    cdbus_Match*  match
    )
{
    if ( NULL != match )
    {
        cdbus_atomicAdd(&match->refCnt, 1);
    }

    return match;
}


void
cdbus_matchUnref
    (
    cdbus_Match* match
    )
{
    cdbus_Int32 value = 0;
    cdbus_Int32 idx;

    if ( NULL != match )
    {
        /* Returns the previous value */
       value = cdbus_atomicSub(&match->refCnt, 1);

       assert( 1 <= value );

       if ( 1 == value )
       {
           cdbus_free(match->rule.member);
           cdbus_free(match->rule.sender);
           cdbus_free(match->rule.objInterface);
           cdbus_free(match->rule.path);
           cdbus_free(match->rule.arg0Namespace);
           cdbus_free(match->rule.localObjPath);
           cdbus_free(match->ruleStr);
           for ( idx = 0; idx < match->nFilterArgs; idx++ )
           {
               cdbus_free(match->rule.filterArgs[idx].value);
           }
           cdbus_free(match->rule.filterArgs);
           cdbus_free(match);
           CDBUS_TRACE((CDBUS_TRC_INFO,
                        "Destroyed the Match instance (%p)", (void*)match));
       }
    }
}


cdbus_Bool
cdbus_matchIsMatch
    (
    cdbus_Match*    match,
    DBusMessage*    msg
    )
{
    cdbus_Bool isMatch = CDBUS_FALSE;
    const cdbus_Char* path;
    const cdbus_Char* value;
    cdbus_UInt16 nArgMatches = 0;
    cdbus_UInt8 maxArgN = 0;
    cdbus_Int32 idx;
    DBusMessageIter dbusIter;
    cdbus_Int32 curDbusType;
    cdbus_Int32 dbusArgIdx = 0;
    cdbus_Int32 lenA;
    cdbus_Int32 lenB;
    cdbus_Int32 dbusMsgType;

    /*
     * For more information on match rules please refer to the D-Bus specification here:
     * http://dbus.freedesktop.org/doc/dbus-specification.html#message-bus-routing-match-rules
     */

    if ( (NULL != match) && (NULL != msg) )
    {
        /* Convert the CDBUS message types to a D-Bus equivalent */
        switch ( match->rule.msgType )
        {
            case CDBUS_MATCH_MSG_SIGNAL:
                dbusMsgType = DBUS_MESSAGE_TYPE_SIGNAL;
                break;

            case CDBUS_MATCH_MSG_METHOD_CALL:
                dbusMsgType = DBUS_MESSAGE_TYPE_METHOD_CALL;
                break;

            case CDBUS_MATCH_MSG_METHOD_RETURN:
                dbusMsgType = DBUS_MESSAGE_TYPE_METHOD_RETURN;
                break;

            case CDBUS_MATCH_MSG_ERROR:
                dbusMsgType = DBUS_MESSAGE_TYPE_ERROR;
                break;

            case CDBUS_MATCH_MSG_ANY:
            default:
                dbusMsgType = DBUS_MESSAGE_TYPE_INVALID;
                break;
        }

        path = dbus_message_get_path(msg);
        isMatch = ((DBUS_MESSAGE_TYPE_INVALID == dbusMsgType) ||
                    (dbusMsgType == dbus_message_get_type(msg))) &&
                  ((match->rule.member == NULL) ? CDBUS_TRUE :
                      dbus_message_has_member(msg, match->rule.member)) &&
                  ((match->rule.sender == NULL) ? CDBUS_TRUE :
                      dbus_message_has_sender(msg, match->rule.sender)) &&
                  ((match->rule.objInterface == NULL) ? CDBUS_TRUE :
                      dbus_message_has_interface(msg, match->rule.objInterface)) &&
                  ((match->rule.path == NULL) ? CDBUS_TRUE :
                      !match->rule.treatPathAsNamespace ?
                          dbus_message_has_path(msg, match->rule.path) :
                          ((NULL == path) ? CDBUS_FALSE :
                              ((0 == strncmp(match->rule.path, path,
                                            strlen(match->rule.path)))
                                               ||
                               ((NULL == match->rule.localObjPath) ? CDBUS_FALSE :
                                   (0 == strncmp(match->rule.path,
                                           match->rule.localObjPath,
                                           strlen(match->rule.path)))))));

        /* Check for a arg0namespace match */
        if ( isMatch && (NULL != match->rule.arg0Namespace) )
        {
            /* Initialization will *only* fail if the message has no arguments */
            if ( !dbus_message_iter_init(msg, &dbusIter) )
            {
                isMatch = CDBUS_FALSE;
            }
            else
            {
                if ( DBUS_TYPE_STRING == dbus_message_iter_get_arg_type(&dbusIter) )
                {
                    dbus_message_iter_get_basic(&dbusIter, &value);
                    isMatch = (0 == strncmp(match->rule.arg0Namespace, value,
                                strlen(match->rule.arg0Namespace)));
                }
            }
        }

        /* Check for argN matches */
        if ( isMatch && (0 < match->nFilterArgs) )
        {
            /* Initialization will *only* fail if the message has no arguments */
            if ( !dbus_message_iter_init(msg, &dbusIter) )
            {
                isMatch = CDBUS_FALSE;
            }
            else
            {
                /* Since the filter args are sorted by increasing argument
                 * index the last one *must* have the largest index.
                 */
                maxArgN = match->rule.filterArgs[match->nFilterArgs - 1].argN;

                /* Iterate over the message arguments while there is a match */
                while ( ((curDbusType = dbus_message_iter_get_arg_type(&dbusIter)) != DBUS_TYPE_INVALID) &&
                    (dbusArgIdx <= maxArgN) && isMatch )
                {
                    /* We can only match on D-Bus string or object path types */
                    if ( (DBUS_TYPE_STRING == curDbusType) || (DBUS_TYPE_OBJECT_PATH == curDbusType) )
                    {
                        value = NULL;
                        dbus_message_iter_get_basic(&dbusIter, &value);
                        for ( idx = 0; (idx < match->nFilterArgs) && isMatch; idx++ )
                        {
                            /* If this is the message argument index we need to test */
                            if ( match->rule.filterArgs[idx].argN == dbusArgIdx )
                            {
                                if ( NULL == match->rule.filterArgs[idx].value )
                                {
                                    /* A NULL filter matches everything */
                                    nArgMatches++;
                                }
                                else if ( NULL == value )
                                {
                                    isMatch = CDBUS_FALSE;
                                    break;
                                }
                                /* Else both the filter value and argument value are non-NULL */
                                else
                                {
                                    /* ArgN matches can *only* match on D-Bus strings */
                                    if ( (CDBUS_FILTER_ARG == match->rule.filterArgs[idx].argType) &&
                                        (DBUS_TYPE_STRING == curDbusType) )
                                    {
                                        if ( 0 == strcmp(match->rule.filterArgs[idx].value, value) )
                                        {
                                            nArgMatches++;
                                        }
                                        else
                                        {
                                            isMatch = CDBUS_FALSE;
                                            break;
                                        }
                                    }
                                    /* Else this must be an argPath rule. It can match on D-Bus strings or
                                     * object path types.
                                     * */
                                    else if ( (CDBUS_FILTER_ARG_PATH == match->rule.filterArgs[idx].argType) )
                                    {
                                        /* If they're identical values then ... */
                                        if ( 0 == strcmp(match->rule.filterArgs[idx].value, value) )
                                        {
                                            nArgMatches++;
                                        }
                                        /* Else one might be a sub-path of the other */
                                        else
                                        {
                                            lenA = strlen(value);
                                            lenB = strlen(match->rule.filterArgs[idx].value);
                                            if ( ((0 < lenA) && ('/' == value[lenA-1]) &&
                                                (0 == strncmp(value, match->rule.filterArgs[idx].value, lenA)))
                                                                    ||
                                                ((0 < lenB) && ('/' == match->rule.filterArgs[idx].value[lenB-1]) &&
                                                (0 == strncmp(match->rule.filterArgs[idx].value, value, lenB))) )
                                            {
                                                nArgMatches++;
                                            }
                                            else
                                            {
                                                isMatch = CDBUS_FALSE;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    dbus_message_iter_next(&dbusIter);
                    dbusArgIdx++;
                }

                /* Now that we've looped through all the message arguments we must make sure
                 * that all the arguments of the filter were matched successfully against
                 * the message arguments.
                 */
                if ( nArgMatches != match->nFilterArgs )
                {
                    isMatch = CDBUS_FALSE;
                }
            }
        }
    }
    return isMatch;
}


void
cdbus_matchDispatch
    (
    struct cdbus_Connection*    conn,
    cdbus_Match*                match,
    DBusMessage*                msg
    )
{
    if ( (NULL != match) && (NULL != msg) )
    {
        if ( NULL != match->handler )
        {
            match->handler(conn, match, msg, match->userData);
        }
    }
}


const cdbus_Char*
cdbus_matchGetRule
    (
    cdbus_Match*  match
    )
{
    cdbus_Char* rule = NULL;
    cdbus_Int32 idx = 0;
    const cdbus_Char* fmt;

    if ( NULL != match )
    {
        if ( NULL != match->ruleStr )
        {
            rule = match->ruleStr;
        }
        else
        {
            cdbus_StringBuffer* sb = cdbus_stringBufferNew(DBUS_MAXIMUM_MATCH_RULE_LENGTH);
            if ( NULL != sb )
            {
                if ( CDBUS_MATCH_MSG_SIGNAL == match->rule.msgType )
                {
                    cdbus_stringBufferAppendFormat(sb, "type='signal'");
                }
                else if ( CDBUS_MATCH_MSG_METHOD_CALL == match->rule.msgType )
                {
                    cdbus_stringBufferAppendFormat(sb, "type='method_call'");
                }
                else if ( CDBUS_MATCH_MSG_METHOD_RETURN == match->rule.msgType )
                {
                    cdbus_stringBufferAppendFormat(sb, "type='method_return'");
                }
                else if ( CDBUS_MATCH_MSG_ERROR == match->rule.msgType )
                {
                    cdbus_stringBufferAppendFormat(sb, "type='error'");
                }

                if ( NULL != match->rule.member )
                {
                    cdbus_stringBufferAppendFormat(sb, "%smember='%s'",
                        cdbus_stringBufferIsEmpty(sb) ? "":",", match->rule.member);
                }

                if ( NULL != match->rule.sender )
                {
                    cdbus_stringBufferAppendFormat(sb, "%ssender='%s'",
                        cdbus_stringBufferIsEmpty(sb) ? "":",", match->rule.sender);
                }

                if ( NULL != match->rule.objInterface )
                {
                    cdbus_stringBufferAppendFormat(sb, "%sinterface='%s'",
                        cdbus_stringBufferIsEmpty(sb) ? "":",", match->rule.objInterface);
                }

                if ( NULL != match->rule.path )
                {
                    if ( match->rule.treatPathAsNamespace )
                    {
                        cdbus_stringBufferAppendFormat(sb, "%spath_namespace='%s'",
                            cdbus_stringBufferIsEmpty(sb) ? "":",", match->rule.path);
                    }
                    else
                    {
                        cdbus_stringBufferAppendFormat(sb, "%spath='%s'",
                            cdbus_stringBufferIsEmpty(sb) ? "":",", match->rule.path);
                    }
                }

#if DBUS_VERSION >= 0x010506
                cdbus_stringBufferAppendFormat(sb, "%seavesdrop='%s'",
                    cdbus_stringBufferIsEmpty(sb) ? "":",",
                    match->rule.eavesdrop ? "true" : "false");
#endif

                for ( idx = 0; idx < match->nFilterArgs; ++idx )
                {
                    if ( NULL != match->rule.filterArgs[idx].value )
                    {
                        if ( CDBUS_FILTER_ARG == match->rule.filterArgs[idx].argType )
                        {
                            if ( cdbus_stringBufferIsEmpty(sb) )
                            {
                                fmt = "arg%u='%s'";
                            }
                            else
                            {
                                fmt = ",arg%u='%s'";
                            }
                        }
                        else if ( CDBUS_FILTER_ARG_PATH == match->rule.filterArgs[idx].argType )
                        {
                            if ( cdbus_stringBufferIsEmpty(sb) )
                            {
                                fmt = "arg%upath='%s'";
                            }
                            else
                            {
                                fmt = ",arg%upath='%s'";
                            }
                        }
                        else
                        {
                            fmt = "";
                        }
                        cdbus_stringBufferAppendFormat(sb, fmt, match->rule.filterArgs[idx].argN,
                                                       match->rule.filterArgs[idx].value);
                    }
                }

                if ( cdbus_stringBufferLength(sb) <= DBUS_MAXIMUM_MATCH_RULE_LENGTH )
                {
                    rule = cdbus_strDup(cdbus_stringBufferRaw(sb));
                    match->ruleStr = rule;
                }
                cdbus_stringBufferUnref(sb);
            }
        }
    }

    return rule;
}


static cdbus_Bool
cdbus_matchModifyFilter
    (
    cdbus_Match*                match,
    struct cdbus_Connection*    conn,
    cdbus_Bool                  add
    )
{
    cdbus_Bool isModified = CDBUS_FALSE;
    DBusError dbusError;
    const cdbus_Char* rule;

    if ( (NULL != match) && (NULL != conn) )
    {
        rule = cdbus_matchGetRule(match);
        if ( NULL != rule )
        {
            dbus_error_init(&dbusError);

            /* These functions *will* block waiting for a reply from the daemon */
            if ( add )
            {
                dbus_bus_add_match(cdbus_connectionGetDBus(conn),
                                   rule,
                                   &dbusError);
            }
            else
            {
                dbus_bus_remove_match(cdbus_connectionGetDBus(conn),
                                      rule,
                                      &dbusError);
            }


            if ( !dbus_error_is_set(&dbusError) )
            {
                isModified = CDBUS_TRUE;
            }
            else
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR,
                             "Failed to modify match: %s", dbusError.message));
            }
            dbus_error_free(&dbusError);
        }
    }

    return isModified;
}


cdbus_Bool
cdbus_matchAddFilter
    (
    cdbus_Match*                match,
    struct cdbus_Connection*    conn
    )
{
    return cdbus_matchModifyFilter(match, conn, CDBUS_TRUE);
}


cdbus_Bool
cdbus_matchRemoveFilter
    (
    cdbus_Match*                match,
    struct cdbus_Connection*    conn
    )
{
    return cdbus_matchModifyFilter(match, conn, CDBUS_FALSE);
}


