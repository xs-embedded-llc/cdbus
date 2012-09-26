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
 * @file           string-pointer-map.c
 * @author         Glenn Schmottlach
 * @brief          Implementation of an string to opaque item registry.
 *******************************************************************************
 */
#include <string.h>
#include <assert.h>
#include "string-pointer-map.h"
#include "atomic-ops.h"
#include "uthash.h"
#include "alloc.h"
#include "mutex.h"


struct cdbus_StrPtrMapNode
{
    cdbus_Char*     key;
    void*           value;
    UT_hash_handle  hh;
};

struct cdbus_StrPtrMap
{
    cdbus_Atomic            refCnt;
    cdbus_StrPtrMapNode*    items;
    cdbus_StrPtrMapFreeFunc freeFunc;
    CDBUS_LOCK_DECLARE(lock);
};


cdbus_StrPtrMap*
cdbus_strPtrMapNew
    (
    cdbus_StrPtrMapFreeFunc  f
    )
{
    cdbus_StrPtrMap* map = cdbus_calloc(1, sizeof(*map));
    if ( NULL != map )
    {
        CDBUS_LOCK_ALLOC(map->lock, CDBUS_MUTEX_RECURSIVE);
        if ( CDBUS_LOCK_IS_NULL(map->lock) )
        {
            cdbus_free(map);
        }
        else
        {
            map->items = NULL;
            map->freeFunc = f;
            map = cdbus_strPtrMapRef(map);
        }
    }
    return map;
}


cdbus_StrPtrMap*
cdbus_strPtrMapRef
    (
    cdbus_StrPtrMap*   map
    )
{
    if ( NULL != map )
    {
        cdbus_atomicAdd(&map->refCnt, 1);
    }

    return map;
}


void
cdbus_strPtrMapUnref
    (
    cdbus_StrPtrMap*   map
    )
{
    cdbus_Int32 value = 0;
    cdbus_StrPtrMapNode* curNode;
    cdbus_StrPtrMapNode* tmpNode;

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
            CDBUS_LOCK_FREE(map->lock);
            cdbus_free(map);
        }
    }
}


cdbus_Bool
cdbus_strPtrMapLock
    (
    cdbus_StrPtrMap*    map
    )
{
    cdbus_Bool isLocked = CDBUS_FALSE;

    if ( NULL != map )
    {
#ifdef CDBUS_ENABLE_THREAD_SUPPORT
        isLocked = CDBUS_LOCK(map->lock);
#else
        isLocked = CDBUS_TRUE;
#endif
    }

    return isLocked;
}


cdbus_Bool
cdbus_strPtrMapUnlock
    (
    cdbus_StrPtrMap*    map
    )
{
    cdbus_Bool isUnlocked = CDBUS_FALSE;

    if ( NULL != map )
    {
#ifdef CDBUS_ENABLE_THREAD_SUPPORT
        isUnlocked = CDBUS_UNLOCK(map->lock);
#else
        isUnlocked = CDBUS_TRUE;
#endif
    }

    return isUnlocked;
}


cdbus_Bool
cdbus_strPtrMapAdd
    (
    cdbus_StrPtrMap*    map,
    cdbus_Char*         key,
    void*               value
    )
{
    cdbus_Bool  isAdded = CDBUS_FALSE;
    cdbus_StrPtrMapNode* node;

    if ( (NULL != map) && (NULL != key) && (NULL != value) )
    {
        /* We can't add items with duplicate keys */
        HASH_FIND_STR(map->items, key, node);

        /* If no match found then ... */
        if ( NULL == node )
        {
            node = cdbus_calloc(1, sizeof(*node));
            if ( NULL != node )
            {
                node->key = key;
                node->value = value;
                HASH_ADD_KEYPTR(hh, map->items, node->key, strlen(node->key), node);
                isAdded = CDBUS_TRUE;
            }
        }
    }

    return isAdded;
}


void*
cdbus_strPtrMapRemove
    (
    cdbus_StrPtrMap*    map,
    const cdbus_Char*   key
    )
{
    void* value = NULL;
    cdbus_StrPtrMapNode* node;

    if ( (NULL != map) && (NULL != key) )
    {
        HASH_FIND_STR(map->items, key, node);
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
cdbus_strPtrMapGet
    (
    cdbus_StrPtrMap*    map,
    const cdbus_Char*   key
    )
{
    void* value = NULL;
    cdbus_StrPtrMapNode* node;

    if ( (NULL != map) && (NULL != key) )
    {
        HASH_FIND_STR(map->items, key, node);
        if ( NULL != node )
        {
            value = node->value;
        }
    }

    return value;
}


cdbus_Bool
cdbus_strPtrMapExists
    (
    cdbus_StrPtrMap*    map,
    const cdbus_Char*   key
    )
{
    cdbus_StrPtrMapNode* node = NULL;

    if ( (NULL != map) && (NULL != key) )
    {
        HASH_FIND_STR(map->items, key, node);
    }

    return node != NULL;
}


cdbus_Bool
cdbus_strPtrMapIterInit
    (
    cdbus_StrPtrMap*        map,
    cdbus_StrPtrMapIter*    iter
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
cdbus_strPtrMapIterIsEnd
    (
    cdbus_StrPtrMapIter*    iter
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
cdbus_strPtrMapIterNext
    (
    cdbus_StrPtrMapIter*    iter
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
cdbus_strPtrMapIterHasNext
    (
    cdbus_StrPtrMapIter*    iter
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
cdbus_strPtrMapIterGet
    (
    cdbus_StrPtrMapIter*    iter,
    cdbus_Char**            key,
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
cdbus_strPtrMapIterRemove
    (
    cdbus_StrPtrMapIter*    iter
    )
{
    cdbus_Bool removed = CDBUS_FALSE;
    cdbus_StrPtrMapNode* nextNode = NULL;

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



