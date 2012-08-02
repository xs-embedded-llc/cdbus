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
 * @file           item-registry.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of a string to opaque item registry.
 *******************************************************************************
 */

#ifndef CDBUS_ITEM_REGISTRY_H_
#define CDBUS_ITEM_REGISTRY_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

typedef struct cdbus_ItemRegistry cdbus_ItemRegistry;
typedef void (*cdbus_ItemFreeFunc)(void*);

cdbus_ItemRegistry* cdbus_itemRegistryNew(cdbus_ItemFreeFunc f);
cdbus_ItemRegistry* cdbus_itemRegistryRef(cdbus_ItemRegistry* reg);
void cdbus_itemRegistryUnref(cdbus_ItemRegistry* reg);
cdbus_Bool cdbus_itemRegistryAdd(cdbus_ItemRegistry* reg, const cdbus_Char* str, void* item);
void* cdbus_itemRegistryRemove(cdbus_ItemRegistry* reg, const cdbus_Char* str);
void* cdbus_itemRegistryGet(cdbus_ItemRegistry* reg, const cdbus_Char* str);
cdbus_Bool cdbus_itemRegistryExists(cdbus_ItemRegistry* reg, const cdbus_Char* str);



CDBUS_END_DECLS


#endif /* Guard for CDBUS_ITEM_REGISTRY_H_ */
