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
 * @file           object-registry.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of a object path to object instance registry.
 *******************************************************************************
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
