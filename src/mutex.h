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
 * @file           mutex.h        
 * @author         Glenn Schmottlach
 * @brief          Generic mutex operations.
 *******************************************************************************
 */

#ifndef CDBUS_MUTEX_H_
#define CDBUS_MUTEX_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"

CDBUS_BEGIN_DECLS

typedef enum
{
    CDBUS_MUTEX_RECURSIVE,
    CDBUS_MUTEX_NORMAL
} cdbus_MutexOption;

typedef struct cdbus_Mutex cdbus_Mutex;

cdbus_Mutex* cdbus_mutexNew(cdbus_MutexOption opt);
void cdbus_mutexFree(cdbus_Mutex* mutex);
cdbus_Bool cdbus_mutexLock(cdbus_Mutex* mutex);
cdbus_Bool cdbus_mutexUnlock(cdbus_Mutex* mutex);
cdbus_Int32 cdbus_mutexCount(cdbus_Mutex* mutex);

#ifdef CDBUS_ENABLE_THREAD_SUPPORT
#define CDBUS_LOCK(X) cdbus_mutexLock(X)
#define CDBUS_UNLOCK(X) cdbus_mutexUnlock(X)
#else
#define CDBUS_LOCK(X)
#define CDBUS_UNLOCK(X)
#endif

CDBUS_END_DECLS

#endif /* Guard for CDBUS_MUTEX_H_ */
