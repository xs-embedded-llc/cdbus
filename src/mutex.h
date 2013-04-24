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
 * @file           mutex.h        
 * @author         Glenn Schmottlach
 * @brief          Generic mutex operations.
 *===========================================================================
 */

#ifndef CDBUS_MUTEX_H_
#define CDBUS_MUTEX_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

typedef enum
{
    CDBUS_MUTEX_RECURSIVE,
    CDBUS_MUTEX_NORMAL
} cdbus_MutexOption;

typedef struct cdbus_Mutex cdbus_Mutex;

cdbus_Mutex* cdbus_mutexNew(cdbus_MutexOption opt);
void cdbus_mutexFree(cdbus_Mutex* mutex);
cdbus_Bool cdbus_mutexLock(cdbus_Mutex* mutex);
cdbus_Bool cdbus_mutexUnlock(cdbus_Mutex* mutex);
cdbus_Int32 cdbus_mutexCount(cdbus_Mutex* mutex);

#ifdef CDBUS_ENABLE_THREAD_SUPPORT
#define CDBUS_LOCK_DECLARE(NAME)    cdbus_Mutex* NAME
#define CDBUS_LOCK_ALLOC(L, OPT)    do { L = cdbus_mutexNew(OPT); } while ( 0 )
#define CDBUS_LOCK_FREE(L)          do { cdbus_mutexFree(L); L = NULL; } while ( 0 )
#define CDBUS_LOCK_IS_NULL(L)       (L == NULL)
#define CDBUS_LOCK(X) cdbus_mutexLock(X)
#define CDBUS_UNLOCK(X) cdbus_mutexUnlock(X)
#else
#define CDBUS_LOCK_DECLARE(NAME)
#define CDBUS_LOCK_ALLOC(L, OPT)
#define CDBUS_LOCK_FREE(L)
#define CDBUS_LOCK_IS_NULL(L)   ( CDBUS_FALSE )
#define CDBUS_LOCK(X)
#define CDBUS_UNLOCK(X)
#endif

CDBUS_END_DECLS

#endif /* Guard for CDBUS_MUTEX_H_ */
