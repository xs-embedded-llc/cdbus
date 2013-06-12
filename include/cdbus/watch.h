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
 * @file           watch.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of a D-Bus watch class.
 *===========================================================================
 */

#ifndef CDBUS_WATCH_H_
#define CDBUS_WATCH_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

/* Forward declarations */
struct cdbus_Dispatcher;
typedef struct cdbus_Watch cdbus_Watch;

/*
 * Note: Currently the return value for the watch handler is unused
 */
typedef cdbus_Bool (*cdbus_WatchHandler)(cdbus_Watch* w,
                                        cdbus_UInt32 rcvEvents,
                                        void* data);

/*
 * Watch flags are defined as bitwise combination of D-Bus flags:
 *      DBUS_WATCH_READABLE;
 *      DBUS_WATCH_WRITABLE;
 *      DBUS_WATCH_ERROR;
 *      DBUS_WATCH_HANGUP
 */
CDBUS_EXPORT cdbus_Watch* cdbus_watchNew(struct cdbus_Dispatcher* dispatcher,
                                     cdbus_Descriptor fd, cdbus_UInt32 flags,
                                     cdbus_WatchHandler h, void* data);
CDBUS_EXPORT cdbus_Watch* cdbus_watchRef(cdbus_Watch* w);
CDBUS_EXPORT void cdbus_watchUnref(cdbus_Watch* w);
CDBUS_EXPORT cdbus_Descriptor cdbus_watchGetDescriptor(cdbus_Watch* w);
CDBUS_EXPORT cdbus_UInt32 cdbus_watchGetFlags(cdbus_Watch* w);
CDBUS_EXPORT cdbus_HResult cdbus_watchSetFlags(cdbus_Watch* w, cdbus_UInt32 flags);
CDBUS_EXPORT cdbus_Bool cdbus_watchIsEnabled(cdbus_Watch* w);
CDBUS_EXPORT cdbus_HResult cdbus_watchEnable(cdbus_Watch* w, cdbus_Bool option);
CDBUS_EXPORT void* cdbus_watchGetData(cdbus_Watch* w);
CDBUS_EXPORT void cdbus_watchSetData(cdbus_Watch* w, void* data);


CDBUS_END_DECLS

#endif /* Guard for CDBUS_WATCH_H_ */
