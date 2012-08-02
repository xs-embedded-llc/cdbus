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
 * @file           semaphore.h
 * @author         Glenn Schmottlach
 * @brief          Generic unnamed semaphore operations.
 *******************************************************************************
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
