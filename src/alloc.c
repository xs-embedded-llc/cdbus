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
 * @file           alloc.c
 * @author         Glenn Schmottlach
 * @brief          Implementation of general allocation functions.
 *******************************************************************************
 */
#include <string.h>
#include "dbus/dbus.h"
#include "alloc.h"

void*
cdbus_malloc
    (
    size_t  size
    )
{
    return dbus_malloc(size);
}


void*
cdbus_calloc
    (
    size_t  nElt,
    size_t  eltSize
    )
{
    return dbus_malloc0(nElt * eltSize);
}


void*
cdbus_realloc
    (
    void*   memory,
    size_t  bytes
    )
{
    return dbus_realloc(memory, bytes);
}


void
cdbus_free
    (
    void*   p
    )
{
    if ( NULL != p )
    {
        dbus_free(p);
    }
}


void cdbus_freeStringArray
    (
    cdbus_Char**    strArray
    )
{
    cdbus_UInt32    idx = 0;

    if ( NULL != strArray )
    {
        while ( NULL != strArray[idx] )
        {
            cdbus_free(strArray[idx]);
        }
        cdbus_free(strArray);
    }
}

cdbus_Char*
cdbus_strDup
    (
    const cdbus_Char*   s
    )
{
    cdbus_Char* p = NULL;

    if ( NULL != s )
    {
        p = cdbus_malloc(strlen(s) + sizeof(cdbus_Char));
        if ( NULL != p )
        {
            p[0] = '\0';
            strcat(p, s);
        }
    }
    return p;
}

