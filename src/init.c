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
 * @file           init.c
 * @author         Glenn Schmottlach
 * @brief          Implementation of the initialization and shutdown functions.
 *******************************************************************************
 */
#include <stddef.h>
#include <cdbus/macros.h>
#include "cdbus/types.h"
#include "cdbus/error.h"
#include "cdbus/dispatcher.h"
#include "dbus/dbus.h"
#include "mutex.h"
#include "pointer-pointer-map.h"
#include "internal.h"
#include "trace.h"


/* Global Variables */
#ifdef CDBUS_ENABLE_THREAD_SUPPORT
CDBUS_LOCK_DECLARE(cdbus_gAtomicOpLock) = NULL;
#endif
cdbus_PtrPtrMap* cdbus_gDispatcherRegistry = NULL;

cdbus_HResult
cdbus_initialize()
{
    cdbus_HResult status = CDBUS_RESULT_SUCCESS;

    /* We initialize "just in case" we want to use this library in a threaded environment */
#ifdef CDBUS_ENABLE_THREAD_SUPPORT
    dbus_threads_init_default();
#endif

    CDBUS_LOCK_ALLOC(cdbus_gAtomicOpLock, CDBUS_MUTEX_RECURSIVE);
    if ( CDBUS_LOCK_IS_NULL(cdbus_gAtomicOpLock) )
    {
        status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                    CDBUS_FAC_CDBUS, CDBUS_EC_ALLOC_FAILURE);
    }
    else
    {
        cdbus_gDispatcherRegistry = cdbus_ptrPtrMapNew(NULL);
        if ( NULL == cdbus_gDispatcherRegistry )
        {
            CDBUS_LOCK_FREE(cdbus_gAtomicOpLock);
            cdbus_gAtomicOpLock = NULL;
            status = CDBUS_MAKE_HRESULT(CDBUS_SEV_FAILURE,
                        CDBUS_FAC_CDBUS, CDBUS_EC_ALLOC_FAILURE);
        }
    }

    return status;
}


cdbus_HResult
cdbus_shutdown()
{
    dbus_shutdown();

    if ( !CDBUS_LOCK_IS_NULL(cdbus_gAtomicOpLock) )
    {
        CDBUS_LOCK_FREE(cdbus_gAtomicOpLock);
    }

    if ( NULL != cdbus_gDispatcherRegistry )
    {
        cdbus_ptrPtrMapUnref(cdbus_gDispatcherRegistry);
    }

    return CDBUS_RESULT_SUCCESS;
}


