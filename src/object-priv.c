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
 * @file           object-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Private implementation of the object class.
 *===========================================================================
 */
#include <string.h>
#include <assert.h>
#include "dbus/dbus.h"
#include "cdbus/error.h"
#include "object-priv.h"
#include "mutex.h"
#include "alloc.h"
#include "trace.h"
#include "internal.h"

#define CDBUS_OBJECT_DEFAULT_INTROSPECT_CAPACITY    (512)


static void
cdbus_objFreeMapItem
    (
    cdbus_Char* name,
    void*       item
    )
{
    CDBUS_UNUSED(name);
    if ( NULL != item )
    {
        cdbus_interfaceUnref((cdbus_Interface*)item);
    }
}


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
            obj->interfaces = cdbus_strPtrMapNew(cdbus_objFreeMapItem);
            CDBUS_LOCK_ALLOC(obj->lock, CDBUS_MUTEX_RECURSIVE);
            obj->objPath = cdbus_strDup(objPath);
            obj->userData = userData;
            obj->handler = defaultHandler;

            if ( !CDBUS_LOCK_IS_NULL(obj->lock) &&
                (NULL != obj->objPath) &&
                (NULL != obj->interfaces))
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
                if ( !CDBUS_LOCK_IS_NULL(obj->lock) )
                {
                    CDBUS_LOCK_FREE(obj->lock);
                }

                if ( NULL != obj->objPath )
                {
                    cdbus_free(obj->objPath);
                }

                if ( NULL != obj->interfaces )
                {
                    cdbus_strPtrMapUnref(obj->interfaces);
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
            cdbus_strPtrMapUnref(obj->interfaces);
            cdbus_free(obj->objPath);
            CDBUS_UNLOCK(obj->lock);
            CDBUS_LOCK_FREE(obj->lock);
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
    cdbus_Bool isAdded = CDBUS_FALSE;
    cdbus_Bool nameExists = CDBUS_FALSE;
    const cdbus_Char* intfName;
    cdbus_Char* curIfName;
    cdbus_Interface* curIntf;
    cdbus_StrPtrMapIter iter;

    if ( (NULL != obj) && (NULL != intf) )
    {
        CDBUS_LOCK(obj->lock);

        intfName = cdbus_interfaceGetName(intf);

        if ( NULL != intfName )
        {
            /* Check to make sure the object doesn't already have an
             * interface with the same name.
             */
            for ( cdbus_strPtrMapIterInit(obj->interfaces, &iter);
                !cdbus_strPtrMapIterIsEnd(&iter);
                cdbus_strPtrMapIterNext(&iter) )
            {
                if ( cdbus_strPtrMapIterGet(&iter, &curIfName,
                    (void**)&curIntf) )
                {
                    /* Check to see if an existing interface already has the
                     * same name.
                     */
                    if ( (NULL != curIfName) &&
                        (0 == strcmp(curIfName, intfName)) )
                    {
                        /* Identical interface names found - can't allow
                         * object to have two interfaces with the same name.
                         */
                        nameExists = CDBUS_TRUE;
                        break;
                    }
                }
            }

            /* If the object doesn't already own an interface with the same
             * name AND it can be added then ...
             */
            if ( !nameExists && cdbus_strPtrMapAdd(obj->interfaces,
                (cdbus_Char*)cdbus_interfaceGetName(intf), intf) )
            {
                cdbus_interfaceRef(intf);
                isAdded = CDBUS_TRUE;
            }
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
    cdbus_Interface* intf;

    if ( (NULL != obj) && (NULL != name) )
    {
        CDBUS_LOCK(obj->lock);
        intf = (cdbus_Interface*)cdbus_strPtrMapRemove(obj->interfaces, name);
        if ( NULL != intf)
        {
            cdbus_interfaceUnref(intf);
            isRemoved = CDBUS_TRUE;
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
            curIntf = (cdbus_Interface*)cdbus_strPtrMapGet(obj->interfaces, intfName);
            if ( NULL != curIntf )
            {
                CDBUS_UNLOCK(obj->lock);
                result = cdbus_interfaceHandleMessage(curIntf, obj, conn, msg);
                CDBUS_LOCK(obj->lock);
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
    cdbus_StrPtrMapIter iter;
    cdbus_Char* intfName;

    if ( NULL != obj )
    {
        sb = cdbus_stringBufferNew(CDBUS_OBJECT_DEFAULT_INTROSPECT_CAPACITY);
        if ( NULL != sb )
        {
            cdbus_stringBufferAppendFormat(sb, "%s",
                                        DBUS_INTROSPECT_1_0_XML_DOCTYPE_DECL_NODE);
            cdbus_stringBufferAppendFormat(sb, "<node name=\"%s\">\n", obj->objPath);
            if ( 0 == strcmp(path, obj->objPath) )
            {
                for ( cdbus_strPtrMapIterInit(obj->interfaces, &iter);
                    !cdbus_strPtrMapIterIsEnd(&iter);
                    cdbus_strPtrMapIterNext(&iter) )
                {
                    if ( cdbus_strPtrMapIterGet(&iter, &intfName, (void**)&intf) )
                    {
                        tmp = cdbus_interfaceIntrospect(intf);
                        if ( NULL != tmp )
                        {
                            cdbus_stringBufferAppend(sb, cdbus_stringBufferRaw(tmp));
                            cdbus_stringBufferUnref(tmp);
                        }
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


