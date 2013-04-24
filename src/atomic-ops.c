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
 * @file           atomic-ops.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of atomic operations.
 *===========================================================================
 */

#include <assert.h>
#include <stddef.h>
#include "atomic-ops.h"
#include "mutex.h"
#include "internal.h"

cdbus_Int32
cdbus_atomicAdd
    (
    cdbus_Atomic*   a,
    cdbus_Int32     v
    )
{
    CDBUS_LOCK(cdbus_gAtomicOpLock);
    assert( NULL != a );
    cdbus_Int32 tmp = a->value;
    a->value += v;
    CDBUS_UNLOCK(cdbus_gAtomicOpLock);
    return tmp;
}


cdbus_Int32
cdbus_atomicSub
    (
    cdbus_Atomic*   a,
    cdbus_Int32     v
    )
{
    CDBUS_LOCK(cdbus_gAtomicOpLock);
    assert( NULL != a );
    cdbus_Int32 tmp = a->value;
    a->value -= v;
    CDBUS_UNLOCK(cdbus_gAtomicOpLock);
    return tmp;
}


cdbus_Int32
cdbus_atomicGet
    (
    cdbus_Atomic*   a
    )
{
    CDBUS_LOCK(cdbus_gAtomicOpLock);
    assert( NULL != a );
    cdbus_Int32 tmp = a->value;
    CDBUS_UNLOCK(cdbus_gAtomicOpLock);
    return tmp;
}


