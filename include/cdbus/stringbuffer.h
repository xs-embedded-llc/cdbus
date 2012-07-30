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
 * @file           stringbuffer.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of a string buffer utility class.
 *******************************************************************************
 */

#ifndef CDBUS_STRINGBUFFER_H_
#define CDBUS_STRINGBUFFER_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

typedef struct cdbus_StringBuffer cdbus_StringBuffer;

CDBUS_EXPORT cdbus_StringBuffer* cdbus_stringBufferNew(cdbus_UInt32 initialCapacity);
CDBUS_EXPORT cdbus_StringBuffer* cdbus_stringBufferCopy(const cdbus_Char* str);
CDBUS_EXPORT cdbus_StringBuffer* cdbus_stringBufferRef(cdbus_StringBuffer* sb);
CDBUS_EXPORT void cdbus_stringBufferUnref(cdbus_StringBuffer* sb);
CDBUS_EXPORT cdbus_UInt32 cdbus_stringBufferAppend(cdbus_StringBuffer* sb,
                                const cdbus_Char* str);
CDBUS_EXPORT cdbus_UInt32 cdbus_stringBufferAppendN(cdbus_StringBuffer* sb,
                                const cdbus_Char* str, cdbus_UInt32 len);
CDBUS_EXPORT cdbus_UInt32 cdbus_stringBufferCapacity(cdbus_StringBuffer* sb);
CDBUS_EXPORT cdbus_UInt32 cdbus_stringBufferLength(cdbus_StringBuffer* sb);
CDBUS_EXPORT const cdbus_Char* cdbus_stringBufferRaw(cdbus_StringBuffer* sb);
CDBUS_EXPORT void cdbus_stringBufferClear(cdbus_StringBuffer* sb);
CDBUS_EXPORT cdbus_Bool cdbus_stringBufferIsEmpty(cdbus_StringBuffer* sb);
CDBUS_EXPORT cdbus_UInt32 cdbus_stringBufferAvailable(cdbus_StringBuffer* sb);
CDBUS_EXPORT cdbus_UInt32 cdbus_stringBufferAppendFormat(cdbus_StringBuffer* sb,
                                                    const cdbus_Char* fmt, ...);


CDBUS_END_DECLS

#endif /* Guard for CDBUS_STRINGBUFFER_H_ */
