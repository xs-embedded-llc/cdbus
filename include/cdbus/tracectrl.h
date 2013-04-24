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
 * @file           tracectrl.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of routines to enable/disable traces.
 *===========================================================================
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
