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
 * @file           object-registry.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of an object path to object registry.
 *******************************************************************************
 */
#include <string.h>
#include <assert.h>
#include "cdbus/object.h"
#include "object-registry.h"
#include "atomic-ops.h"
#include "mutex.h"
#include "uthash.h"
#include "alloc.h"

#define CDBUS_OBJECT_REGISTRY_GROWTH_INC    (5)

typedef struct cdbus_objectRegistryItem
{
    cdbus_Char*     path;       /* key */
    cdbus_Object*   object;     /* value */
    UT_hash_handle  hh;
} cdbus_objectRegistryItem;

struct cdbus_ObjectRegistry
{
    cdbus_Atomic                refCnt;
    CDBUS_LOCK_DECLARE(lock);
    cdbus_objectRegistryItem*   items;
};


cdbus_ObjectRegistry*
cdbus_objectRegistryNew()
{
    cdbus_ObjectRegistry* reg = cdbus_calloc(1, sizeof(*reg));
    if ( NULL != reg )
    {
        CDBUS_LOCK_ALLOC(reg->lock, CDBUS_MUTEX_RECURSIVE);
        if ( CDBUS_LOCK_IS_NULL(reg->lock) )
        {
            cdbus_free(reg);
            reg = NULL;
        }
        else
        {
            reg->items = NULL;
            reg = cdbus_objectRegistryRef(reg);
        }

    }
    return reg;
}


cdbus_ObjectRegistry*
cdbus_objectRegistryRef
    (
    cdbus_ObjectRegistry*   reg
    )
{
    if ( NULL != reg )
    {
        cdbus_atomicAdd(&reg->refCnt, 1);
    }

    return reg;
}


void
cdbus_objectRegistryUnref
    (
    cdbus_ObjectRegistry*   reg
    )
{
    cdbus_Int32 value = 0;
    cdbus_objectRegistryItem* curItem;
    cdbus_objectRegistryItem* tmpItem;

    if ( NULL != reg )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&reg->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            /* Free up the resources */

            CDBUS_LOCK(reg->lock);
            /* Empty out the has table */
            HASH_ITER(hh, reg->items, curItem, tmpItem)
            {
                /* Remove the item from the container */
                HASH_DEL(reg->items, curItem);

                /* It's up to us to actually free it */
                cdbus_free(curItem->path);
                cdbus_objectUnref(curItem->object);
                cdbus_free(curItem);
            }

            CDBUS_UNLOCK(reg->lock);
            CDBUS_LOCK_FREE(reg->lock);
            cdbus_free(reg);
        }
    }
}


cdbus_Bool
cdbus_objectRegistryAdd
    (
    cdbus_ObjectRegistry*   reg,
    const cdbus_Char*       path,
    cdbus_Object*           obj
    )
{
    cdbus_Bool  isAdded = CDBUS_FALSE;
    cdbus_objectRegistryItem* item;

    if ( (NULL != reg) && (NULL != path) && (NULL != obj) )
    {
        /* We can't add objects with duplicate keys (paths) */
        HASH_FIND_STR(reg->items, path, item);

        /* If no match found then ... */
        if ( NULL == item )
        {
            item = cdbus_calloc(1, sizeof(*item));
            if ( NULL != item )
            {
                item->path = cdbus_strDup(path);
                if ( NULL != item->path )
                {
                    item->object = cdbus_objectRef(obj);
                    HASH_ADD_KEYPTR(hh, reg->items, item->path, strlen(item->path), item);
                    isAdded = CDBUS_TRUE;
                }
                else
                {
                    cdbus_free(item);
                }
            }
        }
    }

    return isAdded;
}


cdbus_Object*
cdbus_objectRegistryRemove
    (
    cdbus_ObjectRegistry*   reg,
    const cdbus_Char*       path
    )
{
    cdbus_Object* obj = NULL;
    cdbus_objectRegistryItem* item;

    if ( (NULL != reg) && (NULL != path) )
    {
        HASH_FIND_STR(reg->items, path, item);
        if ( NULL != item )
        {
            HASH_DEL(reg->items, item);
            obj = item->object;
            cdbus_objectUnref(obj);
            cdbus_free(item->path);
            cdbus_free(item);
        }
    }

    return obj;
}


cdbus_Object*
cdbus_objectRegistryGet
    (
    cdbus_ObjectRegistry*   reg,
    const cdbus_Char*       path
    )
{
    cdbus_Object* obj = NULL;
    cdbus_objectRegistryItem* item;

    if ( (NULL != reg) && (NULL != path) )
    {
        HASH_FIND_STR(reg->items, path, item);
        if ( NULL != item )
        {
            obj = item->object;
        }
    }

    return obj;
}


cdbus_Bool
cdbus_objectRegistryExists
    (
    cdbus_ObjectRegistry*   reg,
    const cdbus_Char*       path
    )
{
    cdbus_objectRegistryItem* item = NULL;

    if ( (NULL != reg) && (NULL != path) )
    {
        HASH_FIND_STR(reg->items, path, item);
    }

    return item != NULL;
}


void
cdbus_objectRegistryPathWithMatchingPrefix
    (
    cdbus_ObjectRegistry*   reg,
    const cdbus_Char*       prefix,
    cdbus_Char***           prefixList
    )
{
    cdbus_objectRegistryItem* curItem;
    cdbus_objectRegistryItem* tmpItem;
    cdbus_UInt32 capacity = 0U;
    cdbus_UInt32 idx = 0;
    cdbus_Int32 prefixLen;
    cdbus_Char* buf;

    if ( (NULL != reg) && (NULL != prefix) && (NULL != prefixList) )
    {
        prefixLen = strlen(prefix);
        *prefixList = NULL;

        /* Iterate over all the paths in the hash table */
        HASH_ITER(hh, reg->items, curItem, tmpItem)
        {
            /* If the prefix matches then ... */
            if ( 0 == strncmp(curItem->path, prefix, prefixLen) )
            {
                /* If we don't have at least two slot left to fill then ... */
                if ( idx + 1 >= capacity )
                {
                    buf = cdbus_realloc(*prefixList,
                            CDBUS_OBJECT_REGISTRY_GROWTH_INC *
                            sizeof(cdbus_Char*));
                    if ( NULL == buf )
                    {
                        cdbus_freeStringArray(*prefixList);
                        *prefixList = NULL;
                        break;
                    }
                    else
                    {
                        *prefixList = (cdbus_Char**)buf;
                        capacity += CDBUS_OBJECT_REGISTRY_GROWTH_INC;
                    }
                }

                if ( idx + 1 < capacity )
                {
                    *prefixList[idx] = cdbus_strDup(curItem->path);
                    *prefixList[idx+1] = NULL;
                    if ( NULL == *prefixList[idx] )
                    {
                        cdbus_freeStringArray(*prefixList);
                        *prefixList = NULL;
                        break;
                    }
                    else
                    {
                        idx += 1;
                    }
                }
            }
        }
    }
}


void
cdbus_objectRegistryChildNodesFromPrefix
    (
    cdbus_ObjectRegistry*   reg,
    const cdbus_Char*       prefix,
    cdbus_Char***           children
    )
{
    cdbus_UInt32 idx = 0U;
    cdbus_Int32 prefixLen;
    cdbus_Char* start;
    cdbus_Char* end;
    cdbus_Char* path;

    if ( (NULL != children) && (NULL != prefix) )
    {
        prefixLen = strlen(prefix);
        cdbus_objectRegistryPathWithMatchingPrefix(reg, prefix, children);
        if ( NULL != *children )
        {
            while ( NULL != (*children)[idx] )
            {
                path = (*children)[idx];
                start = &path[prefixLen];
                end =  strchr(start, '/');
                if ( NULL == end )
                {
                    end = start + strlen(start);
                }
                memmove(path, start, end - start);
                path[end - start] = '\0';
                idx += 1;
            }
        }
    }
}


