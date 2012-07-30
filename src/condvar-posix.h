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
 * @file           condvar-posix.h        
 * @author         Glenn Schmottlach
 * @brief          Defines the details of a POSIX-based condvar.
 *******************************************************************************
 */

#ifndef CDBUS_CONDVAR_POSIX_H_
#define CDBUS_CONDVAR_POSIX_H_

#include "cdbus/cdbus.h"
#include <pthread.h>

CDBUS_BEGIN_DECLS

/* Define the POSIX contents of the condvar */
struct cdbus_CondVar
{
    pthread_cond_t  cv;
};

CDBUS_END_DECLS


#endif /* Guard for CDBUS_CONDVAR_POSIX_H_ */
