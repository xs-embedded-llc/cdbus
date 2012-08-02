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
 * @file           item-registry.c
 * @author         Glenn Schmottlach
 * @brief          Implementation of an string to opaque item registry.
 *******************************************************************************
 */
#include <string.h>
#include <assert.h>
#include "item-registry.h"
#include "atomic-ops.h"
#include "mutex.h"
#include "uthash.h"
#include "alloc.h"

#define CDBUS_ITEM_REGISTRY_GROWTH_INC    (5)

typedef struct cdbus_itemRegistryNode
{
    cdbus_Char*     str;      /* key */
    void*           item;     /* value */
    UT_hash_handle  hh;
} cdbus_itemRegistryNode;

struct cdbus_ItemRegistry
{
    cdbus_Atomic            refCnt;
    cdbus_Mutex*            lock;
    cdbus_itemRegistryNode* items;
    cdbus_ItemFreeFunc      freeFunc;
};


cdbus_ItemRegistry*
cdbus_itemRegistryNew
    (
    cdbus_ItemFreeFunc  f
    )
{
    cdbus_ItemRegistry* reg = cdbus_calloc(1, sizeof(*reg));
    if ( NULL != reg )
    {
        reg->lock = cdbus_mutexNew(CDBUS_MUTEX_RECURSIVE);
        if ( NULL == reg->lock )
        {
            cdbus_free(reg);
            reg = NULL;
        }
        else
        {
            reg->items = NULL;
            reg->freeFunc = f;
            reg = cdbus_itemRegistryRef(reg);
        }

    }
    return reg;
}


cdbus_ItemRegistry*
cdbus_itemRegistryRef
    (
    cdbus_ItemRegistry*   reg
    )
{
    if ( NULL != reg )
    {
        cdbus_atomicAdd(&reg->refCnt, 1);
    }

    return reg;
}


void
cdbus_itemRegistryUnref
    (
    cdbus_ItemRegistry*   reg
    )
{
    cdbus_Int32 value = 0;
    cdbus_itemRegistryNode* curNode;
    cdbus_itemRegistryNode* tmpNode;

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
            HASH_ITER(hh, reg->items, curNode, tmpNode)
            {
                /* Remove the item from the container */
                HASH_DEL(reg->items, curNode);

                /* It's up to us to actually free the key */
                cdbus_free(curNode->str);
                if ( NULL != reg->freeFunc )
                {
                    reg->freeFunc(curNode->item);
                }
                cdbus_free(curNode);
            }

            CDBUS_UNLOCK(reg->lock);
            cdbus_mutexFree(reg->lock);
            cdbus_free(reg);
        }
    }
}


cdbus_Bool
cdbus_itemRegistryAdd
    (
    cdbus_ItemRegistry* reg,
    const cdbus_Char*   str,
    void*               item
    )
{
    cdbus_Bool  isAdded = CDBUS_FALSE;
    cdbus_itemRegistryNode* node;

    if ( (NULL != reg) && (NULL != str) && (NULL != item) )
    {
        /* We can't add items with duplicate keys (strs) */
        HASH_FIND_STR(reg->items, str, node);

        /* If no match found then ... */
        if ( NULL == node )
        {
            node = cdbus_calloc(1, sizeof(*node));
            if ( NULL != node )
            {
                node->str = cdbus_strDup(str);
                if ( NULL != node->str )
                {
                    node->item = item;
                    HASH_ADD_KEYPTR(hh, reg->items, node->str, strlen(node->str), node);
                    isAdded = CDBUS_TRUE;
                }
                else
                {
                    cdbus_free(node);
                }
            }
        }
    }

    return isAdded;
}


void*
cdbus_itemRegistryRemove
    (
    cdbus_ItemRegistry* reg,
    const cdbus_Char*   str
    )
{
    void* item = NULL;
    cdbus_itemRegistryNode* node;

    if ( (NULL != reg) && (NULL != str) )
    {
        HASH_FIND_STR(reg->items, str, node);
        if ( NULL != node )
        {
            HASH_DEL(reg->items, node);
            item = node->item;
            cdbus_free(node->str);
            cdbus_free(node);
        }
    }

    return item;
}


void*
cdbus_itemRegistryGet
    (
    cdbus_ItemRegistry* reg,
    const cdbus_Char*   str
    )
{
    void* item = NULL;
    cdbus_itemRegistryNode* node;

    if ( (NULL != reg) && (NULL != str) )
    {
        HASH_FIND_STR(reg->items, str, node);
        if ( NULL != node )
        {
            item = node->item;
        }
    }

    return item;
}


cdbus_Bool
cdbus_itemRegistryExists
    (
    cdbus_ItemRegistry* reg,
    const cdbus_Char*   str
    )
{
    cdbus_itemRegistryNode* node = NULL;

    if ( (NULL != reg) && (NULL != str) )
    {
        HASH_FIND_STR(reg->items, str, node);
    }

    return node != NULL;
}




