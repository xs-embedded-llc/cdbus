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
 * @file           watch-priv.h        
 * @author         Glenn Schmottlach
 * @brief          Private declaration of watch class.
 *===========================================================================
 */

#ifndef CDBUS_WATCH_PRIV_H_
#define CDBUS_WATCH_PRIV_H_

#include "cdbus/types.h"
#include "cdbus/watch.h"
#include "queue.h"
#include "mutex.h"

/* Forward declarations */
struct cdbus_MainLoopWatch;

struct cdbus_Watch
{
    cdbus_Dispatcher*           dispatcher;
    cdbus_Atomic                refCnt;
    struct cdbus_MainLoopWatch* watch;
    void*                       data;
    cdbus_WatchHandler          handler;
    CDBUS_LOCK_DECLARE(lock);
    LIST_ENTRY(cdbus_Watch)     link;
};


#endif /* Guard for CDBUS_WATCH_PRIV_H_ */
