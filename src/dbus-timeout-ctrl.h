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
 * @file           dbus-timeout-ctrl.h
 * @author         Glenn Schmottlach
 * @brief          Declaration of D-Bus related timeout control functions.
 *===========================================================================
 */

#ifndef CDBUS_DBUS_TIMEOUT_CTRL_H_
#define CDBUS_DBUS_TIMEOUT_CTRL_H_

#include "dbus/dbus.h"
#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

dbus_bool_t cdbus_timeoutAddHandler(DBusTimeout* dbusTimeout, void* data);
void cdbus_timeoutRemoveHandler(DBusTimeout* dbusTimeout, void* data);
void cdbus_timeoutToggleHandler(DBusTimeout* dbusTimeout, void* data);

CDBUS_END_DECLS

#endif /* Guard for CDBUS_DBUS_TIMEOUT_CTRL_H_ */
