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
 * @file           internal.h        
 * @author         Glenn Schmottlach
 * @brief          Internal declarations and globals
 *===========================================================================
 */

#ifndef CDBUS_INTERNAL_H_
#define CDBUS_INTERNAL_H_

#include <stddef.h>
#include "cdbus/types.h"
#include "pointer-pointer-map.h"
#include "mutex.h"
#include "ev.h"


/*
 * Global variables
 */

/* Module-wide locks */
extern cdbus_Mutex* cdbus_gAtomicOpLock;
extern cdbus_PtrPtrMap* cdbus_gDispatcherRegistry;


#define CDBUS_UNUSED(X) (void)(X)

#if EV_MULTIPLICITY
#define CDBUS_EV_UNUSED(X) CDBUS_UNUSED(X)
#else
#define CDBUS_EV_UNUSED(X)
#endif

#endif /* Guard for CDBUS_INTERNAL_H_ */
