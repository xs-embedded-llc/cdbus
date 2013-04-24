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
 * @file           object-registry.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of a object path to object instance registry.
 *===========================================================================
 */

#ifndef CDBUS_OBJECT_REGISTRY_H_
#define CDBUS_OBJECT_REGISTRY_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"
#include "cdbus/object.h"

CDBUS_BEGIN_DECLS

typedef struct cdbus_ObjectRegistry cdbus_ObjectRegistry;

cdbus_ObjectRegistry* cdbus_objectRegistryNew();
cdbus_ObjectRegistry* cdbus_objectRegistryRef(cdbus_ObjectRegistry* reg);
void cdbus_objectRegistryUnref(cdbus_ObjectRegistry* reg);
cdbus_Bool cdbus_objectRegistryAdd(cdbus_ObjectRegistry* reg, const cdbus_Char* path, cdbus_Object* obj);
cdbus_Object* cdbus_objectRegistryRemove(cdbus_ObjectRegistry* reg, const cdbus_Char* path);
cdbus_Object* cdbus_objectRegistryGet(cdbus_ObjectRegistry* reg, const cdbus_Char* path);
cdbus_Bool cdbus_objectRegistryExists(cdbus_ObjectRegistry* reg, const cdbus_Char* path);
void cdbus_objectRegistryPathWithMatchingPrefix(cdbus_ObjectRegistry* reg, const cdbus_Char* prefix,
                                                cdbus_Char*** prefixList);
void cdbus_objectRegistryChildNodesFromPrefix(cdbus_ObjectRegistry* reg, const cdbus_Char* prefix,
                                                cdbus_Char*** children);


CDBUS_END_DECLS


#endif /* Guard for CDBUS_OBJECT_REGISTRY_H_ */
