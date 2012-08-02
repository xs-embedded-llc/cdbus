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
 * @file           pointer-pointer-map.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of an opaque pointer to opaque pointer map.
 *******************************************************************************
 */

#ifndef CDBUS_POINTER_POINTER_MAP_H_
#define CDBUS_POINTER_POINTER_MAP_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

typedef struct cdbus_PtrPtrMap cdbus_PtrPtrMap;
typedef struct cdbus_PtrPtrMapNode cdbus_PtrPtrMapNode;
typedef void (*cdbus_PtrPtrMapFreeFunc)(void* key, void* value);
typedef struct cdbus_PtrPtrMapIter
{
    cdbus_PtrPtrMapNode*    node;
    cdbus_PtrPtrMap*        map;
} cdbus_PtrPtrMapIter;

cdbus_PtrPtrMap* cdbus_ptrPtrMapNew(cdbus_PtrPtrMapFreeFunc f);
cdbus_PtrPtrMap* cdbus_ptrPtrMapRef(cdbus_PtrPtrMap* map);
void cdbus_ptrPtrMapUnref(cdbus_PtrPtrMap* map);
cdbus_Bool cdbus_ptrPtrMapLock(cdbus_PtrPtrMap* map);
cdbus_Bool cdbus_ptrPtrMapUnlock(cdbus_PtrPtrMap* map);
cdbus_Bool cdbus_ptrPtrMapAdd(cdbus_PtrPtrMap* map, void* key, void* value);
void* cdbus_ptrPtrMapRemove(cdbus_PtrPtrMap* map, const void* key);
void* cdbus_ptrPtrMapGet(cdbus_PtrPtrMap* map, const void* key);
cdbus_Bool cdbus_ptrPtrMapExists(cdbus_PtrPtrMap* map, const void* key);
cdbus_Bool cdbus_ptrPtrMapIterInit(cdbus_PtrPtrMap* map, cdbus_PtrPtrMapIter* iter);
cdbus_Bool cdbus_ptrPtrMapIterIsEnd(cdbus_PtrPtrMapIter* iter);
cdbus_Bool cdbus_ptrPtrMapIterNext(cdbus_PtrPtrMapIter* iter);
cdbus_Bool cdbus_ptrPtrMapIterHasNext(cdbus_PtrPtrMapIter* iter);
cdbus_Bool cdbus_ptrPtrMapIterGet(cdbus_PtrPtrMapIter* iter, void** key, void** value);
cdbus_Bool cdbus_ptrPtrMapIterRemove(cdbus_PtrPtrMapIter* iter);




CDBUS_END_DECLS


#endif /* Guard for CDBUS_POINTER_POINTER_MAP_H_ */
