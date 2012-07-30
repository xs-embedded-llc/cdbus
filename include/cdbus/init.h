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
 * @file           init.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of initialization and shutdown functions.
 *******************************************************************************
 */

#ifndef CDBUS_INIT_H_
#define CDBUS_INIT_H_

#include "cdbus/types.h"
#include "cdbus/macros.h"

CDBUS_BEGIN_DECLS

CDBUS_EXPORT cdbus_HResult cdbus_initialize();
CDBUS_EXPORT cdbus_HResult cdbus_shutdown();

CDBUS_END_DECLS


#endif /* Guard for CDBUS_INIT_H_ */
