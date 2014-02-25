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
 * @file           init.c
 * @author         Glenn Schmottlach
 * @brief          Implementation of the initialization and shutdown functions.
 *===========================================================================
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

/* Embed library version into module */
const char CDBUS_LIBRARY_VERSION[] = "CDBUS_"CDBUS_VERSION_STRING;

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


