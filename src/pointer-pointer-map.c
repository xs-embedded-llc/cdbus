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
 * @file           pointer-pointer-map.c
 * @author         Glenn Schmottlach
 * @brief          Implementation of an opaque pointer to opaque pointer map.
 *******************************************************************************
 */
#include <string.h>
#include <assert.h>
#include "pointer-pointer-map.h"
#include "atomic-ops.h"
#include "uthash.h"
#include "alloc.h"
#include "mutex.h"


struct cdbus_PtrPtrMapNode
{
    void*           key;
    void*           value;
    UT_hash_handle  hh;
};

struct cdbus_PtrPtrMap
{
    cdbus_Atomic            refCnt;
    cdbus_PtrPtrMapNode*    items;
    cdbus_PtrPtrMapFreeFunc freeFunc;
    cdbus_Mutex*            lock;
};


cdbus_PtrPtrMap*
cdbus_ptrPtrMapNew
    (
    cdbus_PtrPtrMapFreeFunc  f
    )
{
    cdbus_PtrPtrMap* map = cdbus_calloc(1, sizeof(*map));
    if ( NULL != map )
    {
        map->lock = cdbus_mutexNew(CDBUS_MUTEX_RECURSIVE);
        if ( NULL == map->lock )
        {
            cdbus_free(map);
        }
        else
        {
            map->items = NULL;
            map->freeFunc = f;
            map = cdbus_ptrPtrMapRef(map);
        }
    }
    return map;
}


cdbus_PtrPtrMap*
cdbus_ptrPtrMapRef
    (
    cdbus_PtrPtrMap*   map
    )
{
    if ( NULL != map )
    {
        cdbus_atomicAdd(&map->refCnt, 1);
    }

    return map;
}


void
cdbus_ptrPtrMapUnref
    (
    cdbus_PtrPtrMap*   map
    )
{
    cdbus_Int32 value = 0;
    cdbus_PtrPtrMapNode* curNode;
    cdbus_PtrPtrMapNode* tmpNode;

    if ( NULL != map )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&map->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            /* Free up the resources */
            CDBUS_LOCK(map->lock);

            /* Empty out the has table */
            HASH_ITER(hh, map->items, curNode, tmpNode)
            {
                /* Remove the item from the container */
                HASH_DEL(map->items, curNode);

                /* If a function has been provided to free
                 * the key/value data then call it ...
                 */
                if ( NULL != map->freeFunc )
                {
                    map->freeFunc(curNode->key, curNode->value);
                }
                cdbus_free(curNode);
            }

            CDBUS_UNLOCK(map->lock);
            cdbus_mutexFree(map->lock);
            cdbus_free(map);
        }
    }
}


cdbus_Bool
cdbus_ptrPtrMapLock
    (
    cdbus_PtrPtrMap*    map
    )
{
    cdbus_Bool isLocked = CDBUS_FALSE;

    if ( NULL != map )
    {
#ifdef CDBUS_ENABLE_THREAD_SUPPORT
        isLocked = cdbus_mutexLock(map->lock);
#else
        isLocked = CDBUS_TRUE;
#endif
    }

    return isLocked;
}


cdbus_Bool
cdbus_ptrPtrMapUnlock
    (
    cdbus_PtrPtrMap*    map
    )
{
    cdbus_Bool isUnlocked = CDBUS_FALSE;

    if ( NULL != map )
    {
#ifdef CDBUS_ENABLE_THREAD_SUPPORT
        isUnlocked = cdbus_mutexUnlock(map->lock);
#else
        isUnlocked = CDBUS_TRUE;
#endif
    }

    return isUnlocked;
}


cdbus_Bool
cdbus_ptrPtrMapAdd
    (
    cdbus_PtrPtrMap*    map,
    void*               key,
    void*               value
    )
{
    cdbus_Bool  isAdded = CDBUS_FALSE;
    cdbus_PtrPtrMapNode* node;

    if ( (NULL != map) && (NULL != key) && (NULL != value) )
    {
        /* We can't add items with duplicate keys */
        HASH_FIND_PTR(map->items, &key, node);

        /* If no match found then ... */
        if ( NULL == node )
        {
            node = cdbus_calloc(1, sizeof(*node));
            if ( NULL != node )
            {
                node->key = key;
                node->value = value;
                HASH_ADD_PTR(map->items, key, node);
                isAdded = CDBUS_TRUE;
            }
        }
    }

    return isAdded;
}


void*
cdbus_ptrPtrMapRemove
    (
    cdbus_PtrPtrMap*    map,
    const void*         key
    )
{
    void* value = NULL;
    cdbus_PtrPtrMapNode* node;

    if ( (NULL != map) && (NULL != key) )
    {
        HASH_FIND_PTR(map->items, &key, node);
        if ( NULL != node )
        {
            HASH_DEL(map->items, node);
            value = node->value;
            cdbus_free(node);
        }
    }

    return value;
}


void*
cdbus_ptrPtrMapGet
    (
    cdbus_PtrPtrMap*    map,
    const void*         key
    )
{
    void* value = NULL;
    cdbus_PtrPtrMapNode* node;

    if ( (NULL != map) && (NULL != key) )
    {
        HASH_FIND_PTR(map->items, &key, node);
        if ( NULL != node )
        {
            value = node->value;
        }
    }

    return value;
}


cdbus_Bool
cdbus_ptrPtrMapExists
    (
    cdbus_PtrPtrMap*    map,
    const void*         key
    )
{
    cdbus_PtrPtrMapNode* node = NULL;

    if ( (NULL != map) && (NULL != key) )
    {
        HASH_FIND_PTR(map->items, &key, node);
    }

    return node != NULL;
}


cdbus_Bool
cdbus_ptrPtrMapIterInit
    (
    cdbus_PtrPtrMap*        map,
    cdbus_PtrPtrMapIter*    iter
    )
{
    cdbus_Bool isCreated = CDBUS_FALSE;

    if ( (NULL != map) && (NULL != iter) )
    {

        iter->node = map->items;
        iter->map = map;
        isCreated = CDBUS_TRUE;
    }
    return isCreated;
}


cdbus_Bool
cdbus_ptrPtrMapIterIsEnd
    (
    cdbus_PtrPtrMapIter*    iter
    )
{
    cdbus_Bool isEnd = CDBUS_FALSE;

    if ( NULL != iter )
    {
        isEnd = iter->node == NULL;
    }

    return isEnd;
}


cdbus_Bool
cdbus_ptrPtrMapIterNext
    (
    cdbus_PtrPtrMapIter*    iter
    )
{
    cdbus_Bool  hasNext = CDBUS_FALSE;

    if ( (NULL != iter) && (NULL != iter->node) )
    {
        iter->node = iter->node->hh.next;
        hasNext = (iter->node != NULL);
    }
    return hasNext;
}


cdbus_Bool
cdbus_ptrPtrMapIterHasNext
    (
    cdbus_PtrPtrMapIter*    iter
    )
{
    cdbus_Bool hasNext = CDBUS_FALSE;

    if ( (NULL != iter) && (NULL != iter->node) )
    {
        hasNext = (iter->node->hh.next != NULL);
    }
    return hasNext;
}


cdbus_Bool
cdbus_ptrPtrMapIterGet
    (
    cdbus_PtrPtrMapIter*    iter,
    void**                  key,
    void**                  value
    )
{
    cdbus_Bool gotData = CDBUS_FALSE;

    if ( (NULL != iter) && (NULL != key) && (NULL != value) )
    {
        if ( NULL != iter->node )
        {
            *key = iter->node->key;
            *value = iter->node->value;
            gotData = CDBUS_TRUE;
        }
    }

    return gotData;
}


cdbus_Bool
cdbus_ptrPtrMapIterRemove
    (
    cdbus_PtrPtrMapIter*    iter
    )
{
    cdbus_Bool removed = CDBUS_FALSE;
    cdbus_PtrPtrMapNode* nextNode = NULL;

    if ( NULL != iter )
    {
        if ( NULL != iter->node )
        {
            nextNode = iter->node->hh.next;

            if ( NULL != iter->map->freeFunc )
            {
                iter->map->freeFunc(iter->node->key, iter->node->value);
            }
            HASH_DEL(iter->map->items, iter->node);
            cdbus_free(iter->node);
            iter->node = nextNode;
            removed = CDBUS_TRUE;
        }
    }

    return removed;
}



