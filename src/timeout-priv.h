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
 * @file           timeout-priv.h
 * @author         Glenn Schmottlach
 * @brief          Private declaration of timeout class.
 *===========================================================================
 */

#ifndef CDBUS_TIMEOUT_PRIV_H_
#define CDBUS_TIMEOUT_PRIV_H_

#include "cdbus/types.h"
#include "cdbus/timeout.h"
#include "queue.h"
#include "mutex.h"
#include "ev.h"


struct cdbus_Timeout
{
    cdbus_Dispatcher*           dispatcher;
    cdbus_Atomic                refCnt;
    ev_timer                    timerWatcher;
    cdbus_Bool                  repeat;
    cdbus_TimeoutHandler        handler;
    void*                       data;
    CDBUS_LOCK_DECLARE(lock);
    LIST_ENTRY(cdbus_Timeout)   link;
};


#endif /* Guard for CDBUS_TIMEOUT_PRIV_H_ */
