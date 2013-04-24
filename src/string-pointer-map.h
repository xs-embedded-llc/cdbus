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
 * @file           string-pointer-map.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of a string to opaque pointer map.
 *===========================================================================
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
