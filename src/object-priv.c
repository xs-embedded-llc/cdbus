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
 * @file           object-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Private implementation of the object class.
 *******************************************************************************
 */
#include <string.h>
#include <assert.h>
#include "cdbus/error.h"
#include "object-priv.h"
#include "queue.h"
#include "mutex.h"
#include "alloc.h"
#include "trace.h"

#define CDBUS_OBJECT_DEFAULT_INTROSPECT_CAPACITY    (512)
#define CDBUS_INTROSPECT_HEADER "<!DOCTYPE node PUBLIC \"-//freedesktop//DTD D-BUS Object Introspection 1.0//EN\" \"http://www.freedesktop.org/standards/dbus/1.0/introspect.dtd\">"

cdbus_Object*
cdbus_objectNew
    (
    const cdbus_Char*           objPath,
    cdbus_ObjectMessageHandler  defaultHandler,
    void*                       userData
    )
{
    cdbus_Object* obj = NULL;

    if ( NULL != objPath )
    {
        obj = cdbus_calloc(1, sizeof(*obj));
        if ( NULL != obj )
        {
            LIST_INIT(&obj->interfaces);
            obj->lock = cdbus_mutexNew(CDBUS_MUTEX_RECURSIVE);
            obj->objPath = cdbus_strDup(objPath);
            obj->userData = userData;
            obj->handler = defaultHandler;

            if ( (NULL != obj->lock) && (NULL != obj->objPath) )
            {
                obj = cdbus_objectRef(obj);
                CDBUS_TRACE((CDBUS_TRC_INFO,
                      "Created an object instance (%p)", (void*)obj));
            }
            /* Else there was an error allocating the object */
            else
            {
                /* Free up anything that may have already been
                 * allocated.
                 */
                if ( NULL != obj->lock )
                {
                    cdbus_mutexFree(obj->lock);
                }

                if ( NULL != obj->objPath )
                {
                    cdbus_free(obj->objPath);
                }

                cdbus_free(obj);
                obj = NULL;
            }
        }
    }
    return obj;
}


cdbus_Object*
cdbus_objectRef
    (
    cdbus_Object*   obj
    )
{
    if ( NULL != obj )
    {
        cdbus_atomicAdd(&obj->refCnt, 1);
    }

    return obj;
}


void cdbus_objectUnref
    (
    cdbus_Object*   obj
    )
{
    cdbus_Interface* intf = NULL;
    cdbus_Interface* nextIntf = NULL;
    cdbus_Int32 value = 0;

    if ( NULL != obj )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&obj->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            /* Free up the resources */

            CDBUS_LOCK(obj->lock);
            /* Loop through the values and free them */
            for ( intf = LIST_FIRST(&obj->interfaces);
                intf != LIST_END(&obj->interfaces);
                intf = nextIntf )
            {
                nextIntf = LIST_NEXT(intf, link);
                cdbus_interfaceUnref(intf);
            }

            cdbus_free(obj->objPath);
            CDBUS_UNLOCK(obj->lock);
            cdbus_mutexFree(obj->lock);
            cdbus_free(obj);
            CDBUS_TRACE((CDBUS_TRC_INFO,
                                  "Destroyed an object instance (%p)", (void*)obj));
        }
    }
}


const cdbus_Char*
cdbus_objectGetPath
    (
    cdbus_Object*   obj
    )
{
    const cdbus_Char* path = NULL;

    if ( NULL != obj )
    {
        CDBUS_LOCK(obj->lock);
        path = obj->objPath;
        CDBUS_UNLOCK(obj->lock);
    }
    return path;
}


cdbus_HResult
cdbus_objectCopyPath
    (
    cdbus_Object*   obj,
    cdbus_Char*     buf,
    cdbus_UInt32*   size
    )
{
    cdbus_UInt32 available;
    cdbus_HResult rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                            CDBUS_FAC_CDBUS,
                                            CDBUS_EC_INVALID_PARAMETER);
    if ( (NULL != obj) && (NULL != size) )
    {
        CDBUS_LOCK(obj->lock);
        available = *size;
        *size = (strlen(obj->objPath) + 1) * sizeof(cdbus_Char);
        if ( (NULL != buf) && (available >= *size) )
        {
            memcpy(buf, obj->objPath, *size);
            rc = CDBUS_RESULT_SUCCESS;
        }
        else
        {
            rc = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                                    CDBUS_FAC_CDBUS,
                                    CDBUS_EC_INSUFFICIENT_SPACE);
        }
        CDBUS_UNLOCK(obj->lock);
    }

    return rc;
}


void
cdbus_objectSetData
    (
    cdbus_Object*   obj,
    void*           data
    )
{
    if ( NULL != obj )
    {
        CDBUS_LOCK(obj->lock);
        obj->userData = data;
        CDBUS_UNLOCK(obj->lock);
    }
}


void*
cdbus_objectGetData
    (
    cdbus_Object*   obj
    )
{
    void* data = NULL;
    if ( NULL != obj )
    {
        CDBUS_LOCK(obj->lock);
        data = obj->userData;
        CDBUS_UNLOCK(obj->lock);
    }

    return data;
}


cdbus_Bool
cdbus_objectAddInterface
    (
    cdbus_Object*           obj,
    struct cdbus_Interface* intf
    )
{
    cdbus_Bool  isAdded = CDBUS_FALSE;
    cdbus_Interface* curIntf;

    if ( (NULL != obj) && (NULL != intf) )
    {
        CDBUS_LOCK(obj->lock);
        /* Only add the interface if it doesn't already exist in the list */
        LIST_FOREACH(curIntf, &obj->interfaces, link)
        {
            /* If the interface name already exists then ... */
            if ( 0 == strcmp(cdbus_interfaceGetName(intf), cdbus_interfaceGetName(curIntf)) )
            {
                break;
            }
        }

        /* If the interface isn't already in the list of interfaces then ... */
        if ( curIntf == LIST_END(&obj->interfaces) )
        {
            LIST_INSERT_HEAD(&obj->interfaces, intf, link);
            /* The object now holds a reference to the interface */
            cdbus_interfaceRef(intf);
            isAdded = CDBUS_TRUE;
        }
        CDBUS_UNLOCK(obj->lock);
    }

    return isAdded;
}


cdbus_Bool
cdbus_objectRemoveInterface
    (
    cdbus_Object*       obj,
    const cdbus_Char*   name
    )
{
    cdbus_Bool  isRemoved = CDBUS_FALSE;
    cdbus_Interface* curIntf;

    if ( (NULL != obj) && (NULL != name) )
    {
        CDBUS_LOCK(obj->lock);
        /* Only add the interface if it doesn't already exist in the list */
        LIST_FOREACH(curIntf, &obj->interfaces, link)
        {
            /* If the interface name already exists then ... */
            if ( 0 == strcmp(name, cdbus_interfaceGetName(curIntf)) )
            {
                LIST_REMOVE(curIntf, link);
                /* The object now holds a reference to the interface */
                cdbus_interfaceUnref(curIntf);
                isRemoved = CDBUS_TRUE;
                break;
            }
        }

        CDBUS_UNLOCK(obj->lock);
    }

    return isRemoved;
}

DBusHandlerResult
cdbus_objectMessageDispatcher
    (
    cdbus_Object*       obj,
    cdbus_Connection*   conn,
    DBusMessage*        msg
    )
{
    DBusHandlerResult result = DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
    const cdbus_Char* intfName = NULL;
    cdbus_Interface* curIntf;
    cdbus_ObjectMessageHandler defaultHandler;

    if ( (NULL != obj) && (NULL != msg) && (NULL != conn) )
    {
        CDBUS_LOCK(obj->lock);

        /* Let's try to extract the interface from the message if
         * it's present.
         */
        intfName = dbus_message_get_interface(msg);
        if ( NULL != intfName )
        {
            /* Now let's scan the list of registered interfaces looking
             * for a match.
             */

            LIST_FOREACH(curIntf, &obj->interfaces, link)
            {
                /* If the interface name already exists then ... */
                if ( 0 == strcmp(intfName, cdbus_interfaceGetName(curIntf)) )
                {
                    CDBUS_UNLOCK(obj->lock);
                    result = cdbus_interfaceHandleMessage(curIntf, obj, conn, msg);
                    CDBUS_LOCK(obj->lock);
                    break;
                }
            }
        }

        if ( (DBUS_HANDLER_RESULT_NOT_YET_HANDLED == result) &&
            (NULL != obj->handler) )
        {
            defaultHandler = obj->handler;
            CDBUS_UNLOCK(obj->lock);
            result = defaultHandler(obj, conn, msg);
            CDBUS_LOCK(obj->lock);
        }

        CDBUS_UNLOCK(obj->lock);
    }

    return result;
}

cdbus_StringBuffer*
cdbus_objectIntrospect
    (
    cdbus_Object*               obj,
    struct cdbus_Connection*    conn,
    const cdbus_Char*           path
    )
{
    cdbus_StringBuffer* sb = NULL;
    cdbus_StringBuffer* tmp = NULL;
    cdbus_Char** children = NULL;
    cdbus_Interface* intf;
    cdbus_UInt32 idx;

    if ( NULL != obj )
    {
        sb = cdbus_stringBufferNew(CDBUS_OBJECT_DEFAULT_INTROSPECT_CAPACITY);
        if ( NULL != sb )
        {
            cdbus_stringBufferAppendFormat(sb, "%s\n", CDBUS_INTROSPECT_HEADER);
            cdbus_stringBufferAppendFormat(sb, "<node name=\"%s\">\n", obj->objPath);
            if ( 0 == strcmp(path, obj->objPath) )
            {
                LIST_FOREACH(intf, &obj->interfaces, link)
                {
                    tmp = cdbus_interfaceIntrospect(intf);
                    if ( NULL != tmp )
                    {
                        cdbus_stringBufferAppend(sb, cdbus_stringBufferRaw(tmp));
                        cdbus_stringBufferUnref(tmp);
                    }
                }
            }

            if ( dbus_connection_list_registered(cdbus_connectionGetDBus(conn), path, &children) )
            {
                idx = 0;
                while ( (NULL != children) && (NULL != children[idx]) )
                {
                    cdbus_stringBufferAppendFormat(sb, "  <node name=\"%s\"/>\n",
                                                    children[idx]);
                    ++idx;
                }

                if ( NULL != children )
                {
                    dbus_free_string_array(children);
                }
            }

            cdbus_stringBufferAppendFormat(sb, "</node>\n");
        }
    }

    return sb;
}


