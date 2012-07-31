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
 * @file           registry.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of a general purpose registry used to store
 *                 key/value pairs that are both pointers to void.
 *******************************************************************************
 */
#include <stddef.h>
#include <assert.h>
#include "registry.h"
#include "queue.h"
#include "alloc.h"
#include "mutex.h"
#include "atomic-ops.h"

struct cdbus_Registry
{
    cdbus_Atomic                                        refCnt;
    cdbus_Mutex*                                        lock;
    cdbus_RegistryFreeValueFunc                         freeFunc;
    LIST_HEAD(cdbus_RegistryHead, cdbus_RegistryItem)   regItems;
};


typedef struct cdbus_RegistryItem
{
    void*                           key;
    void*                           value;
    LIST_ENTRY(cdbus_RegistryItem)  link;
} cdbus_RegistryItem;


cdbus_Registry*
cdbus_registryNew
    (
    cdbus_RegistryFreeValueFunc f
    )
{
    cdbus_Registry* reg = cdbus_calloc(1, sizeof(*reg));
    if ( NULL != reg )
    {
        LIST_INIT(&reg->regItems);
        reg->freeFunc = f;
        reg->lock = cdbus_mutexNew(CDBUS_MUTEX_RECURSIVE);
        if ( NULL != reg->lock )
        {
            reg = cdbus_registryRef(reg);
        }
        else
        {
            cdbus_free(reg);
            reg = NULL;
        }
    }

    return reg;
}


cdbus_Registry*
cdbus_registryRef
    (
    cdbus_Registry* reg
    )
{
    if ( NULL != reg )
    {
        cdbus_atomicAdd(&reg->refCnt, 1);
    }

    return reg;
}


void
cdbus_registryUnref
    (
    cdbus_Registry* reg
    )
{
    cdbus_RegistryItem* item = NULL;
    cdbus_RegistryItem* nextItem = NULL;
    cdbus_Int32 value = 0;

    if ( NULL != reg )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&reg->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            /* Free up the resources */

            CDBUS_LOCK(reg->lock);
            /* Loop through the values and free them */
            for ( item = LIST_FIRST(&reg->regItems);
                item != LIST_END(reg->regItems);
                item = nextItem )
            {
                nextItem = LIST_NEXT(item, link);
                if ( NULL != reg->freeFunc )
                {
                    reg->freeFunc(item);
                }
                cdbus_free(item);
            }

            CDBUS_UNLOCK(reg->lock);
            cdbus_mutexFree(reg->lock);
            cdbus_free(reg);
        }
    }
}


cdbus_Bool
cdbus_registryAdd
    (
    cdbus_Registry* reg,
    void*           key,
    void*           value
    )
{
    cdbus_Bool  added = CDBUS_FALSE;
    cdbus_RegistryItem* item = NULL;

    if ( NULL != reg )
    {
        /* If the key doesn't already exist in the registry then ... */
        if ( !cdbus_registryExists(reg, key) )
        {
            item = cdbus_calloc(1, sizeof(*item));
            if ( NULL != item )
            {
                item->key = key;
                item->value = value;
                CDBUS_LOCK(reg->lock);
                LIST_INSERT_HEAD(&reg->regItems, item, link);
                added = CDBUS_TRUE;
                CDBUS_UNLOCK(reg->lock);
            }
        }
    }

    return added;
}


void*
cdbus_registryRemove
    (
    cdbus_Registry* reg,
    void*           key
    )
{
    cdbus_RegistryItem* item = NULL;
    void* value = NULL;

    if ( NULL != reg )
    {
        CDBUS_LOCK(reg->lock);
        LIST_FOREACH(item, &reg->regItems, link)
        {
            if ( key == item->key )
            {
                value = item->value;
                LIST_REMOVE(item, link);
                cdbus_free(item);
                break;
            }
        }
        CDBUS_UNLOCK(reg->lock);
    }

    return value;
}


cdbus_Bool
cdbus_registryDestroy
    (
    cdbus_Registry* reg,
    void*           key
    )
{
    cdbus_Bool destroyed = CDBUS_FALSE;
    void* value = NULL;

    if ( cdbus_registryExists(reg, key) )
    {
        value = cdbus_registryRemove(reg, key);
        if ( NULL != reg->freeFunc )
        {
            reg->freeFunc(value);
            destroyed = CDBUS_TRUE;
        }
    }

    return destroyed;
}


void*
cdbus_registryGet
    (
    cdbus_Registry* reg,
    void*           key
    )
{
    cdbus_RegistryItem* item = NULL;
    void* value = NULL;

    if ( NULL != reg )
    {
        CDBUS_LOCK(reg->lock);
        LIST_FOREACH(item, &reg->regItems, link)
        {
            if ( key == item->key )
            {
                value = item->value;
                break;
            }
        }
        CDBUS_UNLOCK(reg->lock);
    }

    return value;
}


cdbus_Bool
cdbus_registryExists
    (
    cdbus_Registry* reg,
    void*           key
    )
{
    cdbus_RegistryItem* item = NULL;
    cdbus_Bool exists = CDBUS_FALSE;

    if ( NULL != reg )
    {
        CDBUS_LOCK(reg->lock);
        LIST_FOREACH(item, &reg->regItems, link)
        {
            if ( key == item->key )
            {
                exists = CDBUS_TRUE;
                break;
            }
        }
        CDBUS_UNLOCK(reg->lock);
    }

    return exists;
}



