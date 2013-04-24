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
 * @file           string-pointer-map.c
 * @author         Glenn Schmottlach
 * @brief          Implementation of an string to opaque item registry.
 *===========================================================================
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



