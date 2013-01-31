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
 * @file           alloc.h
 * @author         Glenn Schmottlach
 * @brief          Declarations of memory allocation routines.
 *******************************************************************************
 */

#ifndef CDBUS_ALLOC_H_
#define CDBUS_ALLOC_H_

#include <stddef.h>
#include "cdbus/types.h"

void* cdbus_malloc(size_t size);
void* cdbus_calloc(size_t numElt, size_t eltSize);
void* cdbus_realloc(void* memory, size_t bytes);
void cdbus_free(void* p);
void cdbus_freeStringArray(cdbus_Char** strArray);
cdbus_Char* cdbus_strDup(const cdbus_Char* s);

#endif /* Guard for CDBUS_ALLOC_H_ */
