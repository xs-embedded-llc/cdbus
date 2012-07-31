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
 * @file           atomic-ops.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of supported atomic operations.
 *******************************************************************************
 */

#ifndef CDBUS_ATOMIC_OPS_H_
#define CDBUS_ATOMIC_OPS_H_

#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

cdbus_Int32 cdbus_atomicAdd(cdbus_Atomic* a, cdbus_Int32 v);
cdbus_Int32 cdbus_atomicSub(cdbus_Atomic* a, cdbus_Int32 v);
cdbus_Int32 cdbus_atomicGet(cdbus_Atomic* a);

CDBUS_END_DECLS

#endif /* Guard for CDBUS_ATOMIC_OPS_H_ */