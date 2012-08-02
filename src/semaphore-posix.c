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
 * @file           semaphore-posix.c
 * @author         Glenn Schmottlach
 * @brief          Posix implementation of a semaphore.
 *******************************************************************************
 */

#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "semaphore.h"
#include "semaphore-posix.h"
#include "trace.h"
#include "alloc.h"

cdbus_Semaphore*
cdbus_semaphoreNew
    (
    cdbus_Int32 initialValue
    )
{
    int rc;
    cdbus_Semaphore* s = NULL;

    s = cdbus_calloc(1, sizeof(*s));
    if ( NULL != s )
    {
        rc = sem_init(&s->sem, 0, initialValue);
        if ( 0 != rc )
        {
            cdbus_free(s);
        }
    }

    return s;
}


void
cdbus_semaphoreFree
    (
    cdbus_Semaphore*    semaphore
    )
{
    if ( NULL != semaphore )
    {
        if ( 0 != sem_destroy(&semaphore->sem) )
        {
            CDBUS_TRACE((CDBUS_TRC_ERROR,
                "Failed to destroy semaphore: %s", strerror(errno)));
        }

        cdbus_free(semaphore);
    }
}


cdbus_Bool
cdbus_semaphorePost
    (
    cdbus_Semaphore*    semaphore
    )
{
    cdbus_Bool posted = CDBUS_FALSE;

    if ( NULL != semaphore )
    {
        posted = (sem_post(&semaphore->sem) == 0);
    }

    return posted;
}


cdbus_Bool
cdbus_semaphoreWait
    (
    cdbus_Semaphore*    semaphore
    )
{
    cdbus_Bool  status = CDBUS_FALSE;

    if ( NULL != semaphore )
    {
        status = (sem_wait(&semaphore->sem) == 0);
    }

    return status;
}


cdbus_Bool
cdbus_semaphoreTryWait
    (
    cdbus_Semaphore*    semaphore
    )
{
    cdbus_Bool  status = CDBUS_FALSE;

    if ( NULL != semaphore )
    {
        status = (sem_trywait(&semaphore->sem) == 0);
    }

    return status;
}


