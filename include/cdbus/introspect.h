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
 * @file           introspect.h        
 * @author         Glenn Schmottlach
 * @brief          The declaration of an introspectable interface.
 *******************************************************************************
 */

#ifndef CDBUS_INTROSPECT_H_
#define CDBUS_INTROSPECT_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"
#include "cdbus/interface.h"

CDBUS_BEGIN_DECLS

CDBUS_EXPORT cdbus_Interface* cdbus_introspectNew();

CDBUS_END_DECLS


#endif /* Guard for CDBUS_INTROSPECT_H_ */
