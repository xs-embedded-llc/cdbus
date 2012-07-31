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
 * @file           dbus-timeout-ctrl.c
 * @author         Glenn Schmottlach
 * @brief          Definition of D-Bus related timeout control functions.
 *******************************************************************************
 */
#include "dbus-timeout-ctrl.h"
#include "trace.h"
#include "alloc.h"
#include "connection-priv.h"
#include "cdbus/timeout.h"
#include "internal.h"

static cdbus_Bool
cdbus_dbusTimeoutHandler
    (
    cdbus_Timeout*  t,
    void *          data
    )
{
    dbus_bool_t handled = TRUE;
    DBusTimeout* dbusTimeout = (DBusTimeout*)data;
    if ( (NULL != t) && (NULL != dbusTimeout) )
    {
        handled = dbus_timeout_handle(dbusTimeout);
    }

    /* If the callback indicates an OOM condition then ... */
    if ( !handled )
    {
        CDBUS_TRACE((CDBUS_TRC_WARN, "D-Bus timeout handler returned OOM"));
        /* Repeat the timeout and try to handle it later */
        cdbus_timeoutSetRepeat(t, CDBUS_TRUE);
    }
    /* Else handler processed timeout "successfully" */
    else
    {
        /* Don't re-start the timer */
        cdbus_timeoutSetRepeat(t, CDBUS_FALSE);
    }
    return handled;
}


static void
cdbus_dbusFreeTimeout
    (
    void*   data
    )
{
    cdbus_Timeout* timeout = (cdbus_Timeout*)data;
    if ( NULL != timeout )
    {
        CDBUS_TRACE((CDBUS_TRC_INFO,
            "Unreferencing timeout associate with D-Bus timeout"));
        cdbus_timeoutUnref(timeout);
    }
}


dbus_bool_t
cdbus_timeoutAddHandler
    (
    DBusTimeout*    dbusTimeout,
    void*           data
    )
{
    dbus_bool_t added = FALSE;
    cdbus_Timeout* tm = NULL;
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    cdbus_Connection* conn = (cdbus_Connection*)data;

    if ( NULL != dbusTimeout )
    {
        tm = cdbus_timeoutNew(conn->dispatcher,
                    dbus_timeout_get_interval(dbusTimeout),
                    CDBUS_FALSE, cdbus_dbusTimeoutHandler,
                    dbusTimeout);
        if ( NULL != tm )
        {
            dbus_timeout_set_data(dbusTimeout, tm, cdbus_dbusFreeTimeout);
            rc = cdbus_timeoutEnable(tm,
                                dbus_timeout_get_enabled(dbusTimeout));
            if ( CDBUS_SUCCEEDED(rc) )
            {
                added = TRUE;
            }
            else
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR,
                                    "Failed to enable timeout (0x%02x)", rc));
            }

            /*
             * On error enabling the timeout the cdbus_dbusFreeTimeout
             * function *should* be called and unref the cdbus timer
             * and thus free it up.
             */
        }
    }
    return added;
}


void
cdbus_timeoutRemoveHandler
    (
    DBusTimeout*    dbusTimeout,
    void*           data
    )
{
    cdbus_Timeout* tm = NULL;
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;
    CDBUS_UNUSED(data);

    if ( NULL != dbusTimeout )
    {
        tm = dbus_timeout_get_data(dbusTimeout);
        if ( NULL != tm )
        {
            /* Disabling the timeout which effectively
             * removes it from timeouts that are being
             * managed.
             */
            rc = cdbus_timeoutEnable(tm, CDBUS_FALSE);
            if ( CDBUS_FAILED(rc) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR,
                    "Failed to disable the timer (0x%02x)", rc));
            }

            /* When the D-Bus timer is destroyed it will also
             * unreference our timeout. No need to explicitly
             * do it here.
             */
        }
    }
}


void
cdbus_timeoutToggleHandler
    (
    DBusTimeout*    dbusTimeout,
    void*           data
    )
{
    cdbus_Connection* conn = (cdbus_Connection*)data;
    cdbus_Timeout* tm = NULL;
    cdbus_HResult rc = CDBUS_RESULT_SUCCESS;

    if ( (NULL != dbusTimeout) && (NULL != conn) )
    {
        tm = (cdbus_Timeout*)dbus_timeout_get_data(dbusTimeout);
        if ( NULL == tm )
        {
            CDBUS_TRACE((CDBUS_TRC_ERROR, "Failed to retrieve CDBUS timeout"));
        }
        else
        {
            rc = cdbus_timeoutEnable(tm, dbus_timeout_get_enabled(dbusTimeout));
            if ( CDBUS_FAILED(rc) )
            {
                CDBUS_TRACE((CDBUS_TRC_ERROR,
                    "Failed to enable timeout (0x%02x)", rc));
            }
        }
    }
}


