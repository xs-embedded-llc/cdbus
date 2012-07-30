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
 * @file           condvar.h        
 * @author         Glenn Schmottlach
 * @brief          Generic conditional variable operations
 *******************************************************************************
 */

#ifndef CDBUS_CONDVAR_H_
#define CDBUS_CONDVAR_H_

#include "cdbus/macros.h"
#include "cdbus/types.h"
#include "mutex.h"

CDBUS_BEGIN_DECLS

typedef struct cdbus_CondVar cdbus_CondVar;

cdbus_CondVar* cdbus_condVarNew();
void cdbus_condVarFree(cdbus_CondVar* cv);
cdbus_Bool cdbus_condVarWait(cdbus_CondVar* cv, cdbus_Mutex* m);
cdbus_Bool cdbus_condVarSignal(cdbus_CondVar* cv);

#ifdef CDBUS_ENABLE_THREAD_SUPPORT
#define CDBUS_CV_WAIT(C,M) cdbus_condVarWait(C,M)
#define CDBUS_CV_SIGNAL(C) cdbus_condVarSignal(C)
#else
#define CDBUS_CV_WAIT(C,M)
#define CDBUS_CV_SIGNAL(C)
#endif

CDBUS_END_DECLS

#endif /* Guard for CDBUS_CONDVAR_H_ */
