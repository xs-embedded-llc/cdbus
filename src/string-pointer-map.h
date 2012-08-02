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
 * @file           string-pointer-map.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of a string to opaque pointer map.
 *******************************************************************************
 */

#ifndef CDBUS_STRING_POINTER_MAP_H_
#define CDBUS_STRING_POINTER_MAP_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

typedef struct cdbus_StrPtrMap cdbus_StrPtrMap;
typedef struct cdbus_StrPtrMapNode cdbus_StrPtrMapNode;
typedef struct cdbus_StrPtrMapIter
{
    cdbus_StrPtrMapNode*    node;
    cdbus_StrPtrMap*        map;
} cdbus_StrPtrMapIter;

typedef void (*cdbus_StrPtrMapFreeFunc)(cdbus_Char* key, void* value);

cdbus_StrPtrMap* cdbus_strPtrMapNew(cdbus_StrPtrMapFreeFunc f);
cdbus_StrPtrMap* cdbus_strPtrMapRef(cdbus_StrPtrMap* map);
void cdbus_strPtrMapUnref(cdbus_StrPtrMap* map);
cdbus_Bool cdbus_strPtrMapLock(cdbus_StrPtrMap* map);
cdbus_Bool cdbus_strPtrMapUnlock(cdbus_StrPtrMap* map);
cdbus_Bool cdbus_strPtrMapAdd(cdbus_StrPtrMap* map, cdbus_Char* key, void* value);
void* cdbus_strPtrMapRemove(cdbus_StrPtrMap* map, const cdbus_Char* key);
void* cdbus_strPtrMapGet(cdbus_StrPtrMap* map, const cdbus_Char* key);
cdbus_Bool cdbus_strPtrMapExists(cdbus_StrPtrMap* map, const cdbus_Char* key);
cdbus_Bool cdbus_strPtrMapIterInit(cdbus_StrPtrMap* map, cdbus_StrPtrMapIter* iter);
cdbus_Bool cdbus_strPtrMapIterIsEnd(cdbus_StrPtrMapIter* iter);
cdbus_Bool cdbus_strPtrMapIterNext(cdbus_StrPtrMapIter* iter);
cdbus_Bool cdbus_strPtrMapIterHasNext(cdbus_StrPtrMapIter* iter);
cdbus_Bool cdbus_strPtrMapIterGet(cdbus_StrPtrMapIter* iter, cdbus_Char** key, void** value);
cdbus_Bool cdbus_strPtrMapIterRemove(cdbus_StrPtrMapIter* iter);




CDBUS_END_DECLS


#endif /* Guard for CDBUS_STRING_POINTER_MAP_H_ */
