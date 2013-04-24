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
 * @file           condvar-posix.c        
 * @author         Glenn Schmottlach
 * @brief          Posix implementation of conditional variable.
 *===========================================================================
 */

#include <pthread.h>
#include "condvar.h"
#include "condvar-posix.h"
#include "mutex-posix.h"
#include "alloc.h"

cdbus_CondVar*
cdbus_condVarNew()
{
    cdbus_CondVar* cv = cdbus_calloc(1, sizeof(*cv));
    if ( NULL != cv )
    {
        /* If there was an error creating the condvar then ... */
        if ( 0 != pthread_cond_init(&cv->cv, NULL) )
        {
            cdbus_free(cv);
            cv = NULL;
        }
    }

    return cv;
}


void
cdbus_condVarFree
    (
    cdbus_CondVar*  cv
    )
{
    if ( NULL != cv )
    {
        pthread_cond_destroy(&cv->cv);
    }
}


struct cdbus_Mux;
cdbus_Bool
cdbus_condVarWait
    (
    cdbus_CondVar*  cv,
    cdbus_Mutex*    m
    )
{
    if ( (NULL == cv) || (NULL == m) )
    {
        return CDBUS_FALSE;
    }
    else
    {
        return pthread_cond_wait(&cv->cv, &m->m) == 0;
    }
}


cdbus_Bool
cdbus_condVarSignal
    (
    cdbus_CondVar*  cv
    )
{
    if ( NULL == cv )
    {
        return CDBUS_FALSE;
    }
    else
    {
        return pthread_cond_signal(&cv->cv) == 0;
    }
}

