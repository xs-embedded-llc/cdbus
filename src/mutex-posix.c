/*******************************************************************************
 *
 * Project         cdbus
 * (c) Copyright   2012 XS-Embedded LLC
 *                 All rights reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *******************************************************************************
 *******************************************************************************
 * @file           mutex-posix.c        
 * @author         Glenn Schmottlach
 * @brief          Posix implementation of a mutex.
 *******************************************************************************
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


