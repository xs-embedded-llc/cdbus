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
 * @file           semaphore-posix.h
 * @author         Glenn Schmottlach
 * @brief          Defines the details of a POSIX-based unnamed semaphore.
 *******************************************************************************
 */

#ifndef CDBUS_SEMAPHORE_POSIX_H_
#define CDBUS_SEMAPHORE_POSIX_H_

#include "cdbus/cdbus.h"
#include <semaphore.h>

CDBUS_BEGIN_DECLS

/* Semaphore definition for posix systems */
struct cdbus_Semaphore
{
    sem_t sem;
};

CDBUS_END_DECLS

#endif /* Guard for CDBUS_SEMAPHORE_POSIX_H_ */
