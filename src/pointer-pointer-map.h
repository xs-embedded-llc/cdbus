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
 * @file           pointer-pointer-map.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of an opaque pointer to opaque pointer map.
 *===========================================================================
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
