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
 * @file           timeout.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of a D-Bus timeout class.
 *===========================================================================
 */

#ifndef CDBUS_TIMEOUT_H_
#define CDBUS_TIMEOUT_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"
#include "dbus/dbus.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct cdbus_Dispatcher;
typedef struct cdbus_Timeout cdbus_Timeout;
typedef cdbus_Bool (*cdbus_TimeoutHandler)(cdbus_Timeout* t, void*);

CDBUS_EXPORT cdbus_Timeout* cdbus_timeoutNew(struct cdbus_Dispatcher* dispatcher,
                                cdbus_Int32 msecInterval, cdbus_Bool repeat,
                                cdbus_TimeoutHandler h, void* data);
CDBUS_EXPORT cdbus_Timeout* cdbus_timeoutRef(cdbus_Timeout* t);
CDBUS_EXPORT void cdbus_timeoutUnref(cdbus_Timeout* t);

CDBUS_EXPORT cdbus_Bool cdbus_timeoutIsEnabled(cdbus_Timeout* t);
CDBUS_EXPORT cdbus_HResult cdbus_timeoutEnable(cdbus_Timeout* t, cdbus_Bool option);
CDBUS_EXPORT cdbus_Int32 cdbus_timeoutInterval(cdbus_Timeout* t);
CDBUS_EXPORT cdbus_HResult cdbus_timeoutSetInterval(cdbus_Timeout* t,
                                                cdbus_Int32 msecInterval);
CDBUS_EXPORT cdbus_Bool cdbus_timeoutGetRepeat(cdbus_Timeout* t);
CDBUS_EXPORT void cdbus_timeoutSetRepeat(cdbus_Timeout* t, cdbus_Bool repeat);
CDBUS_EXPORT void* cdbus_timeoutGetData(cdbus_Timeout* t);
CDBUS_EXPORT void cdbus_timeoutSetData(cdbus_Timeout* t, void* data);



CDBUS_END_DECLS

#endif /* Guard for CDBUS_TIMEOUT_H_ */
