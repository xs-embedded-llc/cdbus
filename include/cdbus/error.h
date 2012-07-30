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
 * @file           error.h        
 * @author         Glenn Schmottlach
 * @brief          Defines error macros and codes for cdbus.
 *******************************************************************************
 */

#ifndef CDBUS_ERROR_H_
#define CDBUS_ERROR_H_

#include "cdbus/types.h"

typedef enum {
    CDBUS_SEV_SUCCESS = 0,
    CDBUS_SEV_FAILURE = 1
} cdbus_Severity;

typedef enum {
    CDBUS_FAC_CDBUS  = 1,
    CDBUS_FAC_DBUS   = 2,
    CDBUS_FAC_EV     = 3,
} cdbus_Facility;

typedef enum {
    CDBUS_EC_OK = 0,
    CDBUS_EC_ALLOC_FAILURE,
    CDBUS_EC_INVALID_PARAMETER,
    CDBUS_EC_NOT_FOUND,
    CDBUS_EC_CONNECTION_OPEN_FAILURE,
    CDBUS_EC_BUS_REG_ERROR,
    CDBUS_EC_INTERNAL
} cdbus_ErrorCode;


#define CDBUS_MAKE_HRESULT(SEV, FAC, CODE) \
        ((cdbus_HResult)( \
            (((SEV) & 0x1U) << 31) | \
            (((1U) & 0x1U) << 29) | \
            (((FAC) & 0x7FFU) << 16) | \
            (((CODE) & 0xFFFFU)) \
        ))

#define CDBUS_SUCCEEDED(R)      ((cdbus_HResult)(R) & (1U << 31))
#define CDBUS_FAILED(R)         (!CDBUS_SUCCEEDED(R))
#define CDBUS_FACILITY(R)       (((cdbus_HResult)(R) >> 16U) & 0x7FFU)
#define CDBUS_ERR_CODE(R)       ((cdbus_HResult)(R) & 0xFFFFU)
#define CDBUS_SEVERITY(R)       (((cdbus_HResult)(R) >> 31U) & 0x1U)
#define CDBUS_RESULT_SUCCESS    CDBUS_MAKE_HRESULT(CDBUS_SEV_SUCCESS, CDBUS_FAC_CDBUS, CDBUS_EC_OK)


#endif /* Guard for CDBUS_ERROR_H_ */
