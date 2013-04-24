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
 * @file           error.h        
 * @author         Glenn Schmottlach
 * @brief          Defines error macros and codes for cdbus.
 *===========================================================================
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
    CDBUS_EC_INSUFFICIENT_SPACE,
    CDBUS_EC_FILTER_ERROR,
    CDBUS_EC_INTERNAL
} cdbus_ErrorCode;


#define CDBUS_MAKE_HRESULT(SEV, FAC, CODE) \
        ((cdbus_HResult)( \
            (((SEV) & 0x1U) << 31) | \
            (((1U) & 0x1U) << 29) | \
            (((FAC) & 0x7FFU) << 16) | \
            (((CODE) & 0xFFFFU)) \
        ))

#define CDBUS_FAILED(R)         ((cdbus_HResult)(R) & (1U << 31))
#define CDBUS_SUCCEEDED(R)      (!CDBUS_FAILED(R))
#define CDBUS_FACILITY(R)       (((cdbus_HResult)(R) >> 16U) & 0x7FFU)
#define CDBUS_ERR_CODE(R)       ((cdbus_HResult)(R) & 0xFFFFU)
#define CDBUS_SEVERITY(R)       (((cdbus_HResult)(R) >> 31U) & 0x1U)
#define CDBUS_RESULT_SUCCESS    CDBUS_MAKE_HRESULT(CDBUS_SEV_SUCCESS, CDBUS_FAC_CDBUS, CDBUS_EC_OK)


#endif /* Guard for CDBUS_ERROR_H_ */
