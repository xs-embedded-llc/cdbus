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
 * @file           condvar-posix.c        
 * @author         Glenn Schmottlach
 * @brief          Posix implementation of conditional variable.
 *******************************************************************************
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

