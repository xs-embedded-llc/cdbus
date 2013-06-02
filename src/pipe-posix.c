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
 * @file           pipe-posix.c
 * @author         Glenn Schmottlach
 * @brief          Posix implementation of a pipe.
 *===========================================================================
 */

#include <unistd.h>
#include <fcntl.h>
#include "pipe.h"
#include "pipe-posix.h"
#include "cdbus/alloc.h"

cdbus_Pipe*
cdbus_pipeNew()
{
    cdbus_Int32 rc;
    cdbus_Pipe* p = cdbus_calloc(1, sizeof(*p));
    if ( NULL != p )
    {
        rc = pipe(p->fd);
        if ( 0 != rc )
        {
            cdbus_free(p);
            p = NULL;
        }
        else
        {
            rc = fcntl(p->fd[0], F_SETFL, O_NONBLOCK);
            if ( 0 == rc ) rc = fcntl(p->fd[1], F_SETFL, O_NONBLOCK);

            if (0 != rc)
            {
                cdbus_pipeDestroy(p);
                p = NULL;
            }
        }
    }

    return p;
}


void
cdbus_pipeDestroy
    (
    cdbus_Pipe*  pipe
    )
{
    if ( NULL != pipe )
    {
        close(pipe->fd[0]);
        close(pipe->fd[1]);
    }
}


void
cdbus_pipeGetFds
    (
    cdbus_Pipe*         pipe,
    cdbus_Descriptor*   readFd,
    cdbus_Descriptor*   writeFd
    )
{
    if ( NULL != pipe )
    {
        if ( NULL != readFd )
        {
            *readFd = pipe->fd[0];
        }

        if ( NULL != writeFd )
        {
            *writeFd = pipe->fd[1];
        }
    }
}


cdbus_Int32
cdbus_pipeRead
    (
    cdbus_Pipe*     pipe,
    void*           buf,
    cdbus_UInt32    count
    )
{
    cdbus_Int32 numRead = -1;

    if ( NULL != pipe )
    {
        numRead = read(pipe->fd[0], buf, count);
    }
    return numRead;
}


cdbus_Int32
cdbus_pipeWrite
    (
    cdbus_Pipe*     pipe,
    const void*     buf,
    cdbus_UInt32    count
    )
{
    cdbus_Int32 numWritten = -1;

    if ( NULL != pipe )
    {
        numWritten = write(pipe->fd[1], buf, count);
    }
    return numWritten;
}

