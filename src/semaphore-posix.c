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
 * @file           semaphore-posix.c
 * @author         Glenn Schmottlach
 * @brief          Posix implementation of a semaphore.
 *===========================================================================
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


