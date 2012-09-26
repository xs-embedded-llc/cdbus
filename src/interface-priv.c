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
 * @file           interface-priv.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of interface class.
 *******************************************************************************
 */
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include "dbus/dbus.h"
#include "interface-priv.h"
#include "alloc.h"
#include "trace.h"

#define CDBUS_INTERFACE_INITIAL_BUF_CAPACITY    (512)

static void
cdbus_interfaceDestroyArg
    (
    cdbus_InterfaceArgs* arg
    )
{
    if ( NULL != arg )
    {
        cdbus_free(arg->name);
        cdbus_free(arg->signature);
    }
}


static void
cdbus_interfaceDestroyItem
    (
    cdbus_InterfaceItem* item
    )
{
    cdbus_UInt32 idx;

    if ( NULL != item )
    {
        cdbus_free(item->name);
        for ( idx = 0; idx < item->nArgs; ++idx )
        {
            cdbus_interfaceDestroyArg(&item->args[idx]);
        }
        cdbus_free(item->args);
    }
}


static void
cdbus_interfaceDestroyProperty
    (
    cdbus_InterfaceProperty* prop
    )
{
    if ( NULL != prop )
    {
        cdbus_free(prop->name);
        cdbus_free(prop->signature);
    }
}


static void
cdbus_interfaceFreeMethodList
    (
    cdbus_Interface* intf
    )
{
    cdbus_InterfaceMethod* curMethod;
    cdbus_InterfaceMethod* nextMethod;
    if ( NULL != intf )
    {
        for ( curMethod = LIST_FIRST(&intf->methods);
            curMethod != LIST_END(&intf->methods);
            curMethod = nextMethod )
        {
            nextMethod = LIST_NEXT(curMethod, link);
            cdbus_interfaceDestroyItem(curMethod);
            cdbus_free(curMethod);
        }
    }
}


static void
cdbus_interfaceFreeSignalList
    (
    cdbus_Interface* intf
    )
{
    cdbus_InterfaceMethod* curSignal;
    cdbus_InterfaceMethod* nextSignal;
    if ( NULL != intf )
    {
        for ( curSignal = LIST_FIRST(&intf->signals);
            curSignal != LIST_END(&intf->signals);
            curSignal = nextSignal )
        {
            nextSignal = LIST_NEXT(curSignal, link);
            cdbus_interfaceDestroyItem(curSignal);
            cdbus_free(curSignal);
        }
    }
}


static void
cdbus_interfaceFreePropertyList
    (
    cdbus_Interface* intf
    )
{
    cdbus_InterfaceProperty* curProp;
    cdbus_InterfaceProperty* nextProp;
    if ( NULL != intf )
    {
        for ( curProp = LIST_FIRST(&intf->props);
            curProp != LIST_END(&intf->props);
            curProp = nextProp )
        {
            nextProp = LIST_NEXT(curProp, link);
            cdbus_interfaceDestroyProperty(curProp);
            cdbus_free(curProp);
        }
    }
}


cdbus_Interface*
cdbus_interfaceNew
    (
    const cdbus_Char*               name,
    cdbus_InterfaceMessageHandler   handler,
    void*                           userData
    )
{
    cdbus_Interface*    intf = NULL;

    if ( (NULL != name) && (strlen(name) <= DBUS_MAXIMUM_NAME_LENGTH) )
    {
        intf = cdbus_calloc(1, sizeof(*intf));
        if ( NULL != intf )
        {
            intf->handler = handler;
            intf->userData = userData;
            intf->name = cdbus_strDup(name);
            LIST_INIT(&intf->methods);
            LIST_INIT(&intf->signals);
            LIST_INIT(&intf->props);
            CDBUS_LOCK_ALLOC(intf->lock, CDBUS_MUTEX_RECURSIVE);
            if ( !CDBUS_LOCK_IS_NULL(intf->lock) && (NULL != intf->name) )
            {
                cdbus_interfaceRef(intf);
                CDBUS_TRACE((CDBUS_TRC_INFO,
                      "Created an interface instance (%p)", (void*)intf));
            }
            else
            {
                /* These functions check for NULL pointers */
                CDBUS_LOCK_FREE(intf->lock);
                cdbus_free(intf->name);
                cdbus_free(intf);
                intf = NULL;
            }
        }
    }

    return intf;
}


cdbus_Interface*
cdbus_interfaceRef
    (
    cdbus_Interface*    intf
    )
{
    if ( NULL != intf )
    {
        cdbus_atomicAdd(&intf->refCnt, 1);
    }

    return intf;
}


void cdbus_interfaceUnref
    (
    cdbus_Interface*    intf
    )
{
    cdbus_Int32 value = 0;

    if ( NULL != intf )
    {
        /* Returns the previous value */
       value = cdbus_atomicSub(&intf->refCnt, 1);

       assert( 1 <= value );

       if ( 1 == value )
       {
           CDBUS_LOCK(intf->lock);
           cdbus_interfaceFreeMethodList(intf);
           cdbus_interfaceFreeSignalList(intf);
           cdbus_interfaceFreePropertyList(intf);

           CDBUS_UNLOCK(intf->lock);
           CDBUS_LOCK_FREE(intf->lock);
           cdbus_free(intf);
           CDBUS_TRACE((CDBUS_TRC_INFO,
                        "Destroyed the interface instance (%p)", (void*)intf));
       }
    }
}


const cdbus_Char*
cdbus_interfaceGetName
    (
    cdbus_Interface*    intf
    )
{
    cdbus_Char* name = NULL;

    if ( NULL != intf )
    {
        CDBUS_LOCK(intf->lock);
        name = intf->name;
        CDBUS_UNLOCK(intf->lock);
    }
    return name;
}


void
cdbus_interfaceSetData
    (
    cdbus_Interface*    intf,
    void*               data
    )
{
    if ( NULL != intf )
    {
        CDBUS_LOCK(intf->lock);
        intf->userData = data;
        CDBUS_UNLOCK(intf->lock);
    }
}


void*
cdbus_interfaceGetData
    (
    cdbus_Interface*    intf
    )
{
    void* userData = NULL;

    if ( NULL != intf )
    {
        CDBUS_LOCK(intf->lock);
        userData = intf->userData;
        CDBUS_UNLOCK(intf->lock);
    }

    return userData;
}


static cdbus_Bool
cdbus_interfaceRegisterItems
    (
    cdbus_Interface*                intf,
    const cdbus_DbusIntrospectItem* methods,
    cdbus_UInt32                    numMethods,
    struct cdbus_InfItemHead*       itemsHead
    )
{
    cdbus_Bool  isRegistered = CDBUS_FALSE;
    cdbus_UInt32 idx;
    cdbus_UInt32 argIdx;
    cdbus_InterfaceItem* item;
    cdbus_InterfaceItem* curItem;
    cdbus_InterfaceItem* nextItem;
    struct cdbus_InfItemHead tmpItems;
    LIST_INIT(&tmpItems);

    if ( (NULL != intf) && (NULL != methods) )
    {
        /* Let's now assume we can register everything */
        isRegistered = CDBUS_TRUE;

        CDBUS_LOCK(intf->lock);
        for ( idx = 0; idx < numMethods; idx++)
        {
            item = cdbus_calloc(1, sizeof(*item));
            if ( NULL == item )
            {
                isRegistered = CDBUS_FALSE;
                break;
            }

            item->name = cdbus_strDup(methods[idx].name);
            if ( NULL == item->name )
            {
                cdbus_interfaceDestroyItem(item);
                cdbus_free(item);
                isRegistered = CDBUS_FALSE;
                break;
            }

            item->args = cdbus_calloc(methods[idx].nArgs, sizeof(*item->args));
            if ( NULL == item->args )
            {
                cdbus_interfaceDestroyItem(item);
                cdbus_free(item);
                isRegistered = CDBUS_FALSE;
                break;
            }

            item->nArgs = 0U;
            for ( argIdx = 0; argIdx < methods[idx].nArgs; argIdx++ )
            {
                item->args[argIdx].xferDir = methods[idx].args[argIdx].xferDir;
                item->args[argIdx].name = cdbus_strDup(methods[idx].args[argIdx].name);
                item->args[argIdx].signature = cdbus_strDup(methods[idx].args[argIdx].signature);
                if ( (NULL == item->args[argIdx].signature) || (NULL == item->args[argIdx].name) )
                {
                    cdbus_interfaceDestroyItem(item);
                    cdbus_free(item);
                    isRegistered = CDBUS_FALSE;
                    break;
                }
                item->nArgs++;
            }

            if ( isRegistered )
            {
                /* Hold these items in a temporary list until
                 * we've processed then all.
                 */
                LIST_INSERT_HEAD(&tmpItems, item, link);
            }
        }


        if ( isRegistered )
        {
            /* Move the methods over from the temporary list
             * to the real list.
             */
            for ( curItem = LIST_FIRST(&tmpItems);
                curItem != LIST_END(&tmpItems);
                curItem = nextItem )
            {
                nextItem = LIST_NEXT(curItem, link);
                LIST_REMOVE(curItem, link);
                LIST_INSERT_HEAD(itemsHead, curItem, link);
            }
        }
        else
        {
            /* Clean up that (partial) temporary list */
            for ( curItem = LIST_FIRST(&tmpItems);
                curItem != LIST_END(&tmpItems);
                curItem = nextItem )
            {
                nextItem = LIST_NEXT(curItem, link);
                cdbus_interfaceDestroyItem(curItem);
                cdbus_free(curItem);
            }
        }

        CDBUS_UNLOCK(intf->lock);
    }
    return isRegistered;
}


cdbus_Bool
cdbus_interfaceRegisterMethods
    (
    cdbus_Interface*                intf,
    const cdbus_DbusIntrospectItem* methods,
    cdbus_UInt32                    numMethods
    )
{
    cdbus_Bool isRegistered = CDBUS_FALSE;

    if ( NULL != intf )
    {
        isRegistered = cdbus_interfaceRegisterItems(intf, methods,
                                        numMethods, &intf->methods);
    }
    return isRegistered;
}


cdbus_Bool
cdbus_interfaceRegisterSignals
    (
    cdbus_Interface*                intf,
    const cdbus_DbusIntrospectItem* signals,
    cdbus_UInt32                    numSignals
    )
{
    cdbus_Bool isRegistered = CDBUS_FALSE;

    if ( NULL != intf )
    {
        isRegistered = cdbus_interfaceRegisterItems(intf, signals,
                                        numSignals, &intf->signals);
    }
    return isRegistered;
}


cdbus_Bool
cdbus_interfaceRegisterProperties
    (
    cdbus_Interface*                    intf,
    const cdbus_DbusIntrospectProperty* properties,
    cdbus_UInt32                        numProperties
    )
{
    cdbus_Bool  isRegistered = CDBUS_FALSE;
    cdbus_UInt32 idx;
    cdbus_InterfaceProperty* prop;
    cdbus_InterfaceProperty* curProp;
    cdbus_InterfaceProperty* nextProp;
    struct cdbus_InfPropHead tmpProps;
    LIST_INIT(&tmpProps);

    if ( (NULL != intf) && (NULL != properties) )
    {
        /* Let's now assume we can register everything */
        isRegistered = CDBUS_TRUE;

        CDBUS_LOCK(intf->lock);
        for ( idx = 0; idx < numProperties; idx++)
        {
            prop = cdbus_calloc(1, sizeof(*prop));
            if ( NULL == prop )
            {
                isRegistered = CDBUS_FALSE;
                break;
            }

            prop->name = cdbus_strDup(properties[idx].name);
            if ( NULL == prop->name )
            {
                cdbus_interfaceDestroyProperty(prop);
                cdbus_free(prop);
                isRegistered = CDBUS_FALSE;
                break;
            }

            prop->signature = cdbus_strDup(properties[idx].signature);
            if ( NULL == prop->signature )
            {
                cdbus_interfaceDestroyProperty(prop);
                cdbus_free(prop);
                isRegistered = CDBUS_FALSE;
                break;
            }

            prop->read = properties[idx].read;
            prop->write = properties[idx].write;

            if ( isRegistered )
            {
                /* Hold these items in a temporary list until
                 * we've processed then all.
                 */
                LIST_INSERT_HEAD(&tmpProps, prop, link);
            }
        }


        if ( isRegistered )
        {
            /* Move the methods over from the temporary list
             * to the real list.
             */
            for ( curProp = LIST_FIRST(&tmpProps);
                curProp != LIST_END(&tmpProps);
                curProp = nextProp )
            {
                nextProp = LIST_NEXT(curProp, link);
                LIST_REMOVE(curProp, link);
                LIST_INSERT_HEAD(&intf->props, curProp, link);
            }
        }
        else
        {
            /* Clean up that (partial) temporary list */
            for ( curProp = LIST_FIRST(&tmpProps);
                curProp != LIST_END(&tmpProps);
                curProp = nextProp )
            {
                nextProp = LIST_NEXT(curProp, link);
                cdbus_interfaceDestroyProperty(curProp);
                cdbus_free(curProp);
            }
        }

        CDBUS_UNLOCK(intf->lock);
    }
    return isRegistered;
}


cdbus_StringBuffer*
cdbus_interfaceIntrospect
    (
    cdbus_Interface*    intf
    )
{
    cdbus_StringBuffer* sb = NULL;
    cdbus_InterfaceItem* item;
    cdbus_InterfaceProperty* prop;
    cdbus_UInt32 idx;

    if ( NULL != intf )
    {
        sb = cdbus_stringBufferNew(CDBUS_INTERFACE_INITIAL_BUF_CAPACITY);
        if ( NULL != sb )
        {
            cdbus_stringBufferAppendFormat(sb, "  <interface name=\"%s\">\n",
                                            intf->name);

            LIST_FOREACH(item, &intf->methods, link)
            {
                cdbus_stringBufferAppendFormat(sb, "    <method name=\"%s\">\n",
                                                item->name);

                for ( idx = 0U; idx < item->nArgs; idx++ )
                {
                    cdbus_stringBufferAppendFormat(sb,
                        "      <arg name=\"%s\" type=\"%s\" direction=\"%s\"/>\n",
                        item->args[idx].name,
                        item->args[idx].signature,
                        (item->args[idx].xferDir == CDBUS_XFER_IN) ? "in":"out"
                        );
                }
                cdbus_stringBufferAppendFormat(sb, "    </method>\n");
            }

            LIST_FOREACH(item, &intf->signals, link)
            {
                cdbus_stringBufferAppendFormat(sb, "    <signal name=\"%s\">\n",
                                                item->name);
                for ( idx = 0U; idx < item->nArgs; idx++ )
                {
                    cdbus_stringBufferAppendFormat(sb,
                        "      <arg name=\"%s\" type=\"%s\" direction=\"out\"/>\n",
                        item->args[idx].name,
                        item->args[idx].signature);
                }
                cdbus_stringBufferAppendFormat(sb, "    </signal>\n");
            }

            LIST_FOREACH(prop, &intf->props, link)
            {
                cdbus_stringBufferAppendFormat(sb,
                    "    <property name=\"%s\" type=\"%s\" access=\"%s%s\"/>\n",
                    prop->name,
                    prop->signature,
                    prop->read ? "read":"",
                    prop->write ? "write":"");
            }

            cdbus_stringBufferAppendFormat(sb, "  </interface>\n");
        }
    }

    return sb;
}


DBusHandlerResult
cdbus_interfaceHandleMessage
    (
    cdbus_Interface*    intf,
    cdbus_Object*       obj,
    cdbus_Connection*   conn,
    DBusMessage*        msg
    )
{
    DBusHandlerResult result = DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
    if ( (NULL != intf) &&
        (NULL != obj) &&
        (NULL != conn) &&
        (NULL != msg) )
    {
        result = intf->handler(conn, obj, msg, intf->userData);
    }


    return result ;
}

