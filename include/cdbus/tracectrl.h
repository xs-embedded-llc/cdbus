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
 * @file           tracectrl.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of routines to enable/disable traces.
 *******************************************************************************
 */

#ifndef CDBUS_TRACECTRL_H_
#define CDBUS_TRACECTRL_H_

#include "cdbus/types.h"

CDBUS_BEGIN_DECLS


#define CDBUS_TRC_OFF   (0)
#define CDBUS_TRC_FATAL (1 << 5)
#define CDBUS_TRC_ERROR (1 << 4)
#define CDBUS_TRC_WARN  (1 << 3)
#define CDBUS_TRC_INFO  (1 << 2)
#define CDBUS_TRC_DEBUG (1 << 1)
#define CDBUS_TRC_TRACE (1 << 0)
#define CDBUS_TRC_ALL   (CDBUS_TRC_FATAL | CDBUS_TRC_ERROR | \
                        CDBUS_TRC_WARN | CDBUS_TRC_INFO | \
                        CDBUS_TRC_DEBUG | CDBUS_TRC_TRACE)

void cdbus_traceSetMask(cdbus_UInt32 mask);
cdbus_UInt32 cdbus_traceGetMask();

CDBUS_END_DECLS

#endif /* Guard for CDBUS_TRACECTRL_H_ */
