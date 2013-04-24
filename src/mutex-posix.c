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
 * @file           mutex-posix.c        
 * @author         Glenn Schmottlach
 * @brief          Posix implementation of a mutex.
 *===========================================================================
 */

/* Makes definition of PTHREAD_MUTEX_RECURSIVE visible under Linux builds */
#define _GNU_SOURCE

#include <stdlib.h>
#include "mutex.h"
#include "mutex-posix.h"
#include "trace.h"
#include "alloc.h"

cdbus_Mutex*
cdbus_mutexNew
    (
    cdbus_MutexOption   opt
    )
{
    int rc;
    cdbus_Mutex* mtx = NULL;

    pthread_mutexattr_t attr;
    rc = pthread_mutexattr_init(&attr);
    if ( 0 == rc )
    {
        if ( CDBUS_MUTEX_RECURSIVE == opt )
        {
            rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        }
        else
        {
            rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_NORMAL);
        }

        if ( 0 == rc )
        {
            mtx = cdbus_malloc(sizeof(*mtx));
            if ( NULL != mtx )
            {
                mtx->count = 0;
                rc = pthread_mutex_init(&mtx->m, &attr);
            }
        }

        if ( (NULL != mtx) && (0 != rc) )
        {
            cdbus_free(mtx);
            mtx = 0;
        }
        rc = pthread_mutexattr_destroy(&attr);
    }

    return mtx;
}


void
cdbus_mutexFree
    (
    cdbus_Mutex*    mutex
    )
{
    if ( NULL != mutex )
    {
        pthread_mutex_destroy(&mutex->m);
    }
}


cdbus_Bool
cdbus_mutexLock
    (
    cdbus_Mutex*    mutex
    )
{
    cdbus_Bool locked = CDBUS_FALSE;

    if ( NULL != mutex )
    {
        locked = (pthread_mutex_lock(&mutex->m) == 0);
        if ( locked )
        {
            mutex->count++;
        }
    }

    return locked;
}


cdbus_Bool
cdbus_mutexUnlock
    (
    cdbus_Mutex*    mutex
    )
{
    cdbus_Bool unlocked = CDBUS_FALSE;

    if ( NULL != mutex )
    {
        unlocked = (pthread_mutex_unlock(&mutex->m) == 0);
        if ( unlocked )
        {
            mutex->count--;
        }
    }

    return unlocked;
}


cdbus_Int32
cdbus_mutexCount
    (
    cdbus_Mutex*    mutex
    )
{
    if ( NULL != mutex )
    {
        return mutex->count;
    }
    else
    {
        return 0;
    }
}


