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
 * @file           pipe.h
 * @author         Glenn Schmottlach
 * @brief          Generic pipe abstraction.
 *===========================================================================
 */

#ifndef CDBUS_PIPE_H_
#define CDBUS_PIPE_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

typedef struct cdbus_Pipe cdbus_Pipe;

cdbus_Pipe* cdbus_pipeNew();
void cdbus_pipeDestroy(cdbus_Pipe* pipe);
void cdbus_pipeGetFds(cdbus_Pipe* pipe, cdbus_Descriptor* readFd,
                        cdbus_Descriptor* writeFd);
cdbus_Int32 cdbus_pipeRead(cdbus_Pipe* pipe, void* buf, cdbus_UInt32 count);
cdbus_Int32 cdbus_pipeWrite(cdbus_Pipe* pipe, const void* buf,
                            cdbus_UInt32 count);

CDBUS_END_DECLS

#endif /* Guard for CDBUS_PIPE_H_ */
