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
 * @file           stringbuffer.c        
 * @author         Glenn Schmottlach
 * @brief          Implementation of the string buffer utility class.
 *===========================================================================
 */
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include "cdbus/stringbuffer.h"
#include "cdbus/alloc.h"
#include "stringbuffer-priv.h"

#define CDBUS_STRINGBUFFER_GROWTH_FACTOR    (1.25)
#define CDBUS_STRINGBUFFER_DEFAULT_SIZE     (128)

static cdbus_UInt32
cdbus_growIfNeeded
    (
    cdbus_StringBuffer* sb,
    cdbus_UInt32        needed
    )
{
    cdbus_UInt32 newCapacity = 0;
    cdbus_Char* newBuf;
    cdbus_UInt32 available;

    if ( NULL != sb )
    {
        available = cdbus_stringBufferAvailable(sb);
        if ( available < needed )
        {
            newCapacity = needed - available + sb->capacity;
            newBuf = cdbus_realloc(sb->buf, newCapacity);
            if ( NULL == newBuf )
            {
                newCapacity = 0U;
            }
            else
            {
                sb->buf = newBuf;
                sb->capacity = newCapacity;
            }
        }
    }

    return cdbus_stringBufferAvailable(sb);
}


cdbus_StringBuffer*
cdbus_stringBufferNew
    (
    cdbus_UInt32    initialCapacity
    )
{
    cdbus_StringBuffer* sb = cdbus_calloc(1, sizeof(*sb));
    if ( NULL != sb )
    {
        if ( initialCapacity == 0 )
        {
            initialCapacity = CDBUS_STRINGBUFFER_DEFAULT_SIZE;
        }

        if ( initialCapacity != cdbus_growIfNeeded(sb, initialCapacity) )
        {
            cdbus_free(sb->buf);
            cdbus_free(sb);
            sb = NULL;
        }
        else
        {
            sb = cdbus_stringBufferRef(sb);
            sb->buf[0] = '\0';
        }
    }

    return sb;
}


cdbus_StringBuffer*
cdbus_stringBufferCopy
    (
    const cdbus_Char*   str
    )
{
    cdbus_UInt32    capacity = 0;

    cdbus_StringBuffer* sb = NULL;
    if ( NULL != str )
    {
        capacity = strlen(str) + sizeof(cdbus_Char);
    }

    sb = cdbus_stringBufferNew(capacity);
    if ( NULL != sb )
    {
        cdbus_stringBufferAppend(sb, str);
    }
    return sb;
}


cdbus_StringBuffer*
cdbus_stringBufferRef
    (
    cdbus_StringBuffer* sb
    )
{
    if ( NULL != sb )
    {
        cdbus_atomicAdd(&sb->refCnt, 1);
    }

    return sb;
}


void
cdbus_stringBufferUnref
    (
    cdbus_StringBuffer* sb
    )
{
    cdbus_Int32 value = 0;

    if ( NULL != sb )
    {
        /* Returns the previous value */
        value = cdbus_atomicSub(&sb->refCnt, 1);

        assert( 1 <= value );

        if ( 1 == value )
        {
            cdbus_free(sb->buf);
            cdbus_free(sb);
        }
    }
}


cdbus_UInt32
cdbus_stringBufferAppend
    (
    cdbus_StringBuffer* sb,
    const cdbus_Char*   str
    )
{
    cdbus_UInt32 nAppended = 0U;

    if ( (NULL != sb) && (NULL != str) )
    {
        nAppended = cdbus_stringBufferAppendN(sb, str, strlen(str));
    }

    return nAppended;
}


cdbus_UInt32
cdbus_stringBufferAppendN
    (
    cdbus_StringBuffer* sb,
    const cdbus_Char*   str,
    cdbus_UInt32        len
    )
{
    cdbus_UInt32 nAppended = 0;
    cdbus_UInt32 needed = 0;

    if ( (NULL != sb) && (NULL != str) )
    {
        needed = len + sizeof(cdbus_Char);
        if ( needed > cdbus_stringBufferAvailable(sb) )
        {
            cdbus_growIfNeeded(sb,
                needed * CDBUS_STRINGBUFFER_GROWTH_FACTOR);

        }

        if ( needed <= cdbus_stringBufferAvailable(sb) )
        {
            strncat(&sb->buf[sb->length], str, len);
            sb->length += len;
            nAppended = len;
        }
    }

    return nAppended;
}


cdbus_UInt32
cdbus_stringBufferCapacity
    (
    cdbus_StringBuffer* sb
    )
{
    return (sb == NULL) ? 0U : sb->capacity;
}


cdbus_UInt32
cdbus_stringBufferLength
    (
    cdbus_StringBuffer* sb
    )
{
    return (sb == NULL) ? 0U : sb->length;
}


cdbus_UInt32
cdbus_stringBufferAvailable
    (
    cdbus_StringBuffer* sb
    )
{
    return ( NULL == sb ) ? 0 : sb->capacity - sb->length;
}


const cdbus_Char*
cdbus_stringBufferRaw
    (
    cdbus_StringBuffer* sb
    )
{
    return ( sb == NULL ) ? NULL : sb->buf;
}


void
cdbus_stringBufferClear
    (
    cdbus_StringBuffer* sb
    )
{
    if ( NULL != sb )
    {
        sb->length = 0;
        if ( sb->capacity > 0U )
        {
            assert( NULL != sb->buf );
            sb->buf[0] = '\0';
        }
    }
}


cdbus_Bool
cdbus_stringBufferIsEmpty
    (
    cdbus_StringBuffer* sb
    )
{
    return ( NULL == sb ) ? CDBUS_TRUE : sb->length == 0U;
}


cdbus_UInt32
cdbus_stringBufferAppendFormat
    (
    cdbus_StringBuffer* sb,
    const cdbus_Char*   fmt,
    ...
    )
{
    va_list args;
    cdbus_Int32 nWritten = 0;
    cdbus_UInt32 available;
    cdbus_UInt32 newAvail;

    if ( (NULL != sb) && (NULL != fmt) )
    {
        while ( CDBUS_TRUE )
        {
            available = cdbus_stringBufferAvailable(sb);
            va_start(args, fmt);
            nWritten = vsnprintf(&sb->buf[sb->length], available, fmt, args);
            va_end(args);

            if ( (nWritten > -1) && (cdbus_UInt32)nWritten < available )
            {
                sb->length += nWritten;
                sb->buf[sb->length] = '\0';
                break;
            }
            else
            {
                /* Try to grow the buffer again to accommodate the string */

                /* If we're compiled with glibc 2.1 */
                if ( nWritten > -1 )
                {
                    newAvail = cdbus_growIfNeeded(sb,
                        (nWritten + sizeof(cdbus_Char)) *
                         CDBUS_STRINGBUFFER_GROWTH_FACTOR);
                }
                /* Else glibc 2.0 behavior */
                else
                {
                    newAvail = cdbus_growIfNeeded(sb, 2 * sb->capacity);
                }

                /* If we can't allocate more memory then ... */
                if ( newAvail == available )
                {
                    nWritten = 0;
                    break;
                }
            }
        }
    }

    return nWritten;
}


