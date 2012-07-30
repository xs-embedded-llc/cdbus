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
 * @file           registry.h        
 * @author         Glenn Schmottlach
 * @brief          General purpose registry used to store key/value pairs that
 *                 are both pointers to void.
 *******************************************************************************
 */

#ifndef CDBUS_REGISTRY_H_
#define CDBUS_REGISTRY_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

typedef struct cdbus_Registry cdbus_Registry;
typedef void (*cdbus_RegistryFreeValueFunc)(void*);

cdbus_Registry* cdbus_registryNew(cdbus_RegistryFreeValueFunc f);
cdbus_Registry* cdbus_registryRef(cdbus_Registry* reg);
void cdbus_registryUnref(cdbus_Registry* reg);
cdbus_Bool cdbus_registryAdd(cdbus_Registry* reg, void* key, void* value);
void* cdbus_registryRemove(cdbus_Registry* reg, void* key);
cdbus_Bool cdbus_registryDestroy(cdbus_Registry* reg, void* key);
void* cdbus_registryGet(cdbus_Registry* reg, void* key);
cdbus_Bool cdbus_registryExists(cdbus_Registry* reg, void* key);


CDBUS_END_DECLS

#endif /* Guard for CDBUS_REGISTRY_H_ */
