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
 * @file           signal-match.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of an object representing a signal match and
 *                 the operations on it.
 *******************************************************************************
 */
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include "signal-match.h"
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


cdbus_SignalMatch*
cdbus_signalMatchNew
    (
    cdbus_connectionSignalHandler   handler,
    void*                           userData,
    const cdbus_SignalRule*         rule
    )
{
    cdbus_SignalMatch* obj = NULL;
    cdbus_Int32 idx;
    cdbus_Bool allocError = CDBUS_FALSE;

    if ( (NULL != handler) && (NULL != rule) )
    {
        obj = cdbus_calloc(1, sizeof(*obj));
        if ( NULL != obj )
        {
            obj->handler = handler;
            obj->userData = userData;
            obj->rule.signalName = cdbus_strDup(rule->signalName);
            obj->rule.sender = cdbus_strDup(rule->sender);
            obj->rule.objInterface = cdbus_strDup(rule->objInterface);
            obj->rule.path = cdbus_strDup(rule->path);
            obj->rule.arg0Namespace = cdbus_strDup(rule->arg0Namespace);
            obj->rule.treatPathAsNamespace = rule->treatPathAsNamespace;
            if ( 0 < rule->nFilterArgs )
            {
                obj->rule.filterArgs = cdbus_calloc(rule->nFilterArgs, sizeof(*(obj->rule.filterArgs)));
                if ( NULL == obj->rule.filterArgs )
                {
                    allocError = CDBUS_TRUE;
                }
                else
                {
                    for ( idx = 0; idx < rule->nFilterArgs; ++idx )
                    {
                        obj->rule.filterArgs[idx].argType = rule->filterArgs[idx].argType;
                        obj->rule.filterArgs[idx].argN = rule->filterArgs[idx].argN;
                        obj->rule.filterArgs[idx].value = cdbus_strDup(rule->filterArgs[idx].value);
                    }
                    obj->rule.nFilterArgs = rule->nFilterArgs;

                    /* Sort these in order of increasing argument index */
                    qsort(obj->rule.filterArgs,
                        obj->rule.nFilterArgs, sizeof(*obj->rule.filterArgs), compareArgs);
                }
            }

            obj = cdbus_signalMatchRef(obj);
            if ( allocError )
            {
                cdbus_signalMatchUnref(obj);
                obj = NULL;
            }
        }
    }
    return obj;
}


cdbus_SignalMatch*
cdbus_signalMatchRef
    (
    cdbus_SignalMatch*  sigMatch
    )
{
    if ( NULL != sigMatch )
    {
        cdbus_atomicAdd(&sigMatch->refCnt, 1);
    }

    return sigMatch;
}


void
cdbus_signalMatchUnref
    (
    cdbus_SignalMatch* sigMatch
    )
{
    cdbus_Int32 value = 0;
    cdbus_Int32 idx;

    if ( NULL != sigMatch )
    {
        /* Returns the previous value */
       value = cdbus_atomicSub(&sigMatch->refCnt, 1);

       assert( 1 <= value );

       if ( 1 == value )
       {
           cdbus_free(sigMatch->rule.signalName);
           cdbus_free(sigMatch->rule.sender);
           cdbus_free(sigMatch->rule.objInterface);
           cdbus_free(sigMatch->rule.path);
           cdbus_free(sigMatch->rule.arg0Namespace);
           cdbus_free(sigMatch->ruleStr);
           for ( idx = 0; idx < sigMatch->rule.nFilterArgs; idx++ )
           {
               cdbus_free(sigMatch->rule.filterArgs[idx].value);
           }
           cdbus_free(sigMatch->rule.filterArgs);
           cdbus_free(sigMatch);
           CDBUS_TRACE((CDBUS_TRC_INFO,
                        "Destroyed the SignalMatch instance (%p)", (void*)sigMatch));
       }
    }
}


cdbus_Bool
cdbus_signalMatchIsMatch
    (
    cdbus_SignalMatch*  sigMatch,
    DBusMessage*        msg
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

    if ( (NULL != sigMatch) && (NULL != msg) )
    {
        path = dbus_message_get_path(msg);
        isMatch = (DBUS_MESSAGE_TYPE_SIGNAL == dbus_message_get_type(msg)) &&
                  ((sigMatch->rule.signalName == NULL) ? CDBUS_TRUE :
                      dbus_message_has_member(msg, sigMatch->rule.signalName)) &&
                  ((sigMatch->rule.sender == NULL) ? CDBUS_TRUE :
                      dbus_message_has_sender(msg, sigMatch->rule.sender)) &&
                  ((sigMatch->rule.objInterface == NULL) ? CDBUS_TRUE :
                      dbus_message_has_interface(msg, sigMatch->rule.objInterface)) &&
                  ((sigMatch->rule.path == NULL) ? CDBUS_TRUE :
                      sigMatch->rule.treatPathAsNamespace ?
                      ((NULL == path) ?
                          CDBUS_FALSE :
                          (0 == strncmp(sigMatch->rule.path, path,
                                        strlen(sigMatch->rule.path)))) :
                      dbus_message_has_path(msg, sigMatch->rule.path));

        if ( isMatch && (0 < sigMatch->rule.nFilterArgs) )
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
                maxArgN = sigMatch->rule.filterArgs[sigMatch->rule.nFilterArgs - 1].argN;

                /* Iterate over the message arguments while there is a match */
                while ( ((curDbusType = dbus_message_iter_get_arg_type(&dbusIter)) != DBUS_TYPE_INVALID) &&
                    (dbusArgIdx <= maxArgN) && isMatch )
                {
                    /* We can only match on D-Bus string or object path types */
                    if ( (DBUS_TYPE_STRING == curDbusType) || (DBUS_TYPE_OBJECT_PATH == curDbusType) )
                    {
                        value = NULL;
                        dbus_message_iter_get_basic(&dbusIter, &value);
                        for ( idx = 0; (idx < sigMatch->rule.nFilterArgs) && isMatch; idx++ )
                        {
                            /* If this is the message argument index we need to test */
                            if ( sigMatch->rule.filterArgs[idx].argN == dbusArgIdx )
                            {
                                if ( NULL == sigMatch->rule.filterArgs[idx].value )
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
                                    if ( (CDBUS_FILTER_ARG == sigMatch->rule.filterArgs[idx].argType) &&
                                        (DBUS_TYPE_STRING == curDbusType) )
                                    {
                                        if ( 0 == strcmp(sigMatch->rule.filterArgs[idx].value, value) )
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
                                    else if ( (CDBUS_FILTER_ARG_PATH == sigMatch->rule.filterArgs[idx].argType) )
                                    {
                                        /* If they're identical values then ... */
                                        if ( 0 == strcmp(sigMatch->rule.filterArgs[idx].value, value) )
                                        {
                                            nArgMatches++;
                                        }
                                        /* Else one might be a sub-path of the other */
                                        else
                                        {
                                            lenA = strlen(value);
                                            lenB = strlen(sigMatch->rule.filterArgs[idx].value);
                                            if ( (0 == strncmp(value, sigMatch->rule.filterArgs[idx].value, lenA)) ||
                                                (0 == strncmp(sigMatch->rule.filterArgs[idx].value, value, lenB)) )
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
                 * that all the arguments if the filter were matched successfully against
                 * the message arguments.
                 */
                if ( nArgMatches != sigMatch->rule.nFilterArgs )
                {
                    isMatch = CDBUS_FALSE;
                }
            }
        }
    }
    return isMatch;
}


void
cdbus_signalMatchDispatch
    (
    struct cdbus_Connection*    conn,
    cdbus_SignalMatch*          sigMatch,
    DBusMessage*                msg
    )
{
    if ( (NULL != sigMatch) && (NULL != msg) )
    {
        if ( NULL != sigMatch->handler )
        {
            sigMatch->handler(conn, sigMatch, msg, sigMatch->userData);
        }
    }
}


const cdbus_Char*
cdbus_signalMatchGetRule
    (
    cdbus_SignalMatch*  sigMatch
    )
{
    cdbus_Char* rule = NULL;
    cdbus_Int32 idx = 0;
    const cdbus_Char* fmt;

    if ( NULL != sigMatch )
    {
        if ( NULL != sigMatch->ruleStr )
        {
            rule = sigMatch->ruleStr;
        }
        else
        {
            cdbus_StringBuffer* sb = cdbus_stringBufferNew(DBUS_MAXIMUM_MATCH_RULE_LENGTH);
            if ( NULL != sb )
            {
                cdbus_stringBufferAppendFormat(sb, "type='signal'");
                if ( NULL != sigMatch->rule.signalName )
                {
                    cdbus_stringBufferAppendFormat(sb, ",member='%s'", sigMatch->rule.signalName);
                }

                if ( NULL != sigMatch->rule.sender )
                {
                    cdbus_stringBufferAppendFormat(sb, ",sender='%s'", sigMatch->rule.sender);
                }

                if ( NULL != sigMatch->rule.objInterface )
                {
                    cdbus_stringBufferAppendFormat(sb, ",interface='%s'", sigMatch->rule.objInterface);
                }

                if ( NULL != sigMatch->rule.path )
                {
                    if ( sigMatch->rule.treatPathAsNamespace )
                    {
                        cdbus_stringBufferAppendFormat(sb, ",path='%s'", sigMatch->rule.path);
                    }
                    else
                    {
                        cdbus_stringBufferAppendFormat(sb, ",path_namespace='%s'", sigMatch->rule.path);
                    }
                }

                for ( idx = 0; idx < sigMatch->rule.nFilterArgs; ++idx )
                {
                    if ( NULL != sigMatch->rule.filterArgs[idx].value )
                    {
                        if ( CDBUS_FILTER_ARG == sigMatch->rule.filterArgs[idx].argType )
                        {
                            fmt = ",arg%u='%s'";
                        }
                        else if ( CDBUS_FILTER_ARG_PATH == sigMatch->rule.filterArgs[idx].argType )
                        {
                            fmt = ",arg%upath='%s'";
                        }
                        else
                        {
                            fmt = "";
                        }
                        cdbus_stringBufferAppendFormat(sb, fmt, sigMatch->rule.filterArgs[idx].argN,
                                                       sigMatch->rule.filterArgs[idx].value);
                    }
                }

                if ( cdbus_stringBufferLength(sb) <= DBUS_MAXIMUM_MATCH_RULE_LENGTH )
                {
                    rule = cdbus_strDup(cdbus_stringBufferRaw(sb));
                    sigMatch->ruleStr = rule;
                }
                cdbus_stringBufferUnref(sb);
            }
        }
    }

    return rule;
}


static cdbus_Bool
cdbus_signalMatchModifyFilter
    (
    cdbus_SignalMatch*          sigMatch,
    struct cdbus_Connection*    conn,
    cdbus_Bool                  add
    )
{
    cdbus_Bool isModified = CDBUS_FALSE;
    DBusError dbusError;
    const cdbus_Char* rule;

    if ( (NULL != sigMatch) && (NULL != conn) )
    {
        rule = cdbus_signalMatchGetRule(sigMatch);
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
cdbus_signalMatchAddFilter
    (
    cdbus_SignalMatch*          sigMatch,
    struct cdbus_Connection*    conn
    )
{
    return cdbus_signalMatchModifyFilter(sigMatch, conn, CDBUS_TRUE);
}


cdbus_Bool
cdbus_signalMatchRemoveFilter
    (
    cdbus_SignalMatch*          sigMatch,
    struct cdbus_Connection*    conn
    )
{
    return cdbus_signalMatchModifyFilter(sigMatch, conn, CDBUS_FALSE);
}


