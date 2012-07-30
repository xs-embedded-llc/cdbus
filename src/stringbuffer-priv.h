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
 * @file           stringbuffer-priv.h        
 * @author         Glenn Schmottlach
 * @brief          Private declaration of the string buffer utility class.
 *******************************************************************************
 */

#ifndef CDBUS_STRINGBUFFER_PRIV_H_
#define CDBUS_STRINGBUFFER_PRIV_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"
#include "atomic-ops.h"

CDBUS_BEGIN_DECLS

struct cdbus_StringBuffer
{
    cdbus_Atomic    refCnt;
    cdbus_Char*     buf;
    cdbus_UInt32    length;
    cdbus_UInt32    capacity;
};

CDBUS_END_DECLS

#endif /* Guard for CDBUS_STRINGBUFFER_PRIV_H_ */
