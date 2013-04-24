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
 * @file           semaphore.h
 * @author         Glenn Schmottlach
 * @brief          Generic unnamed semaphore operations.
 *===========================================================================
 */

#ifndef CDBUS_SEMAPHORE_H_
#define CDBUS_SEMAPHORE_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

typedef struct cdbus_Semaphore cdbus_Semaphore;

cdbus_Semaphore* cdbus_semaphoreNew(cdbus_Int32 initialValue);
void cdbus_semaphoreFree(cdbus_Semaphore* semaphore);
cdbus_Bool cdbus_semaphorePost(cdbus_Semaphore* semaphore);
cdbus_Bool cdbus_semaphoreWait(cdbus_Semaphore* semaphore);
cdbus_Bool cdbus_semaphoreTryWait(cdbus_Semaphore* semaphore);

#ifdef CDBUS_ENABLE_THREAD_SUPPORT
#define CDBUS_SEM_POST(X) cdbus_semaphorePost(X)
#define CDBUS_SEM_WAIT(X) cdbus_semaphoreWait(X)
#define CDBUS_SEM_TRY_WAIT(X) cdbus_semaphoreTryWait(X)
#else
#define CDBUS_SEM_POST(X)
#define CDBUS_SEM_WAIT(X)
#define CDBUS_SEM_TRY_WAIT(X)
#endif

CDBUS_END_DECLS

#endif /* Guard for CDBUS_SEMAPHORE_H_ */
