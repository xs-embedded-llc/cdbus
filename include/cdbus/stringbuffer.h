/*===========================================================================
 * 
 * Project         cdbus
 *
 * Released under the MIT License (MIT)
 * Copyright (c) 2013 XS-Embedded LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
 * NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *===========================================================================
 *===========================================================================
 * @file           stringbuffer.h        
 * @author         Glenn Schmottlach
 * @brief          Declaration of a string buffer utility class.
 *===========================================================================
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
