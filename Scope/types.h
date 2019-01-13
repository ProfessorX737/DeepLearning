//#pragma once
#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <tuple>
#include <cstdlib>
#include "logging.h"
#include <iostream>

typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

typedef uint32 uint;

enum DataType {
	DT_INVALID = 0,
	DT_FLOAT = 1,
	DT_DOUBLE = 2,
	DT_INT64 = 3,
	DT_INT32 = 4,
	DT_INT16 = 5,
	DT_INT8 = 6,
	DT_UINT64 = 7,
	DT_UINT32 = 8,
	DT_UINT16 = 9,
	DT_UINT8 = 10,
	DT_STRING = 11,
	DT_BOOL = 12
};

template <class T> 
struct DataTypeToEnum {
	static constexpr bool valid = false;
	static_assert(valid, "Not a valid type");
};

template<DataType T>
struct EnumToDataType {
	static constexpr bool valid = false;
	static_assert(valid, "No real type for Enum");
};


#define TYPE_AND_ENUM(TYPE,ENUM)				\
	template<>									\
	struct DataTypeToEnum<TYPE> {				\
		static constexpr bool valid = true;		\
		static DataType v() { return ENUM; }	\
		static constexpr DataType value = ENUM;	\
	};											\
	template<>									\
	struct EnumToDataType<ENUM> {				\
		static constexpr bool valid = true;		\
		typedef TYPE type;						\
    };


TYPE_AND_ENUM(float, DT_FLOAT);
TYPE_AND_ENUM(double, DT_DOUBLE);
TYPE_AND_ENUM(int64, DT_INT64);
TYPE_AND_ENUM(int32, DT_INT32);
TYPE_AND_ENUM(int16, DT_INT16);
TYPE_AND_ENUM(int8, DT_INT8);
TYPE_AND_ENUM(uint64, DT_UINT64);
TYPE_AND_ENUM(uint32, DT_UINT32);
TYPE_AND_ENUM(uint16, DT_UINT16);
TYPE_AND_ENUM(uint8, DT_UINT8);
TYPE_AND_ENUM(std::string, DT_STRING);
TYPE_AND_ENUM(bool, DT_BOOL);

#undef TYPE_AND_ENUM

//#define CALL_float(m, ...) m(float, __VA_ARGS__)
//#define CALL_double(m, ...) m(double, __VA_ARGS__)
//#define CALL_int64(m, ...) m(int64, __VA_ARGS__)
//#define CALL_int32(m, ...) m(int32, __VA_ARGS__)
//#define CALL_int16(m, ...) m(int16, __VA_ARGS__)
//#define CALL_int8(m, ...) m(int8, __VA_ARGS__)
//#define CALL_uint64(m, ...) m(uint64, __VA_ARGS__)
//#define CALL_uint32(m, ...) m(uint32, __VA_ARGS__)
//#define CALL_uint16(m, ...) m(uint16, __VA_ARGS__)
//#define CALL_uint8(m, ...) m(uint8, __VA_ARGS__)
//#define CALL_string(m, ...) m(std::string, __VA_ARGS__)
//#define CALL_bool(m, ...) m(bool, __VA_ARGS__)

#define USING_FLOATING_POINT_TYPES

#if defined(USING_DT_FLOAT) || defined(USING_ALL_TYPES) || defined(USING_FLOATING_POINT_TYPES)
#define CALL_float(m, ...) m(float, __VA_ARGS__)
#else
#define CALL_float(m, ...)
#endif
#if defined(USING_DT_DOUBLE) || defined(USING_ALL_TYPES) || defined(USING_FLOATING_POINT_TYPES)
#define CALL_double(m, ...) m(double, __VA_ARGS__)
#else
#define CALL_double(m, ...)
#endif
#if defined(USING_DT_INT64) || defined(USING_ALL_TYPES)
#define CALL_int64(m, ...) m(int64, __VA_ARGS__)
#else
#define CALL_int64(m, ...)
#endif
#if defined(USING_DT_INT32) || defined(USING_ALL_TYPES)
#define CALL_int32(m, ...) m(int32, __VA_ARGS__)
#else
#define CALL_int32(m, ...)
#endif
#if defined(USING_DT_INT16) || defined(USING_ALL_TYPES)
#define CALL_int16(m, ...) m(int16, __VA_ARGS__)
#else
#define CALL_int16(m, ...)
#endif
#if defined(USING_DT_INT8) || defined(USING_ALL_TYPES)
#define CALL_int8(m, ...) m(int8, __VA_ARGS__)
#else
#define CALL_int8(m, ...)
#endif
#if defined(USING_DT_UINT64) || defined(USING_ALL_TYPES)
#define CALL_uint64(m, ...) m(uint64, __VA_ARGS__)
#else
#define CALL_uint64(m, ...)
#endif
#if defined(USING_DT_UINT32) || defined(USING_ALL_TYPES)
#define CALL_uint32(m, ...) m(uint32, __VA_ARGS__)
#else
#define CALL_uint32(m, ...)
#endif
#if defined(USING_DT_UINT16) || defined(USING_ALL_TYPES)
#define CALL_uint16(m, ...) m(uint16, __VA_ARGS__)
#else
#define CALL_uint16(m, ...)
#endif
#if defined(USING_DT_UINT8) || defined(USING_ALL_TYPES)
#define CALL_uint8(m, ...) m(uint8, __VA_ARGS__)
#else
#define CALL_uint8(m, ...)
#endif
#if defined(USING_DT_STRING) || defined(USING_ALL_TYPES)
#define CALL_string(m, ...) m(std::string, __VA_ARGS__)
#else
#define CALL_string(m, ...)
#endif
#if defined(USING_DT_BOOL) || defined(USING_ALL_TYPES)
#define CALL_bool(m, ...) m(bool, __VA_ARGS__)
#else
#define CALL_bool(m, ...)
#endif

#define CALL_INTEGRALS(m, ...)					\
CALL_int64(m, __VA_ARGS__)						\
CALL_int32(m, __VA_ARGS__)						\
CALL_int16(m, __VA_ARGS__)						\
CALL_int8(m, __VA_ARGS__)						\
CALL_uint64(m, __VA_ARGS__)						\
CALL_uint32(m, __VA_ARGS__)						\
CALL_uint16(m, __VA_ARGS__)						\
CALL_uint8(m, __VA_ARGS__)


#define CALL_NUMBER_TYPES(m, ...)				\
CALL_float(m, __VA_ARGS__)						\
CALL_double(m, __VA_ARGS__)						\
CALL_INTEGRALS(m, __VA_ARGS__)	

#define CALL_POD_TYPES(m, ...)					\
CALL_NUMBER_TYPES(m, __VA_ARGS__)				\
CALL_bool(m, __VA_ARGS__)

#define CALL_ALL_TYPES(m, ...)					\
CALL_POD_TYPES(m, __VA_ARGS__);					\
CALL_string(m, __VA_ARGS__)

struct DataTypeSize {
//    static int v(DataType dt) {
////    #define CASE(T,...) if(DataTypeToEnum<T>::v() == dt) { return sizeof(T); }
////    	CALL_NUMBER_TYPES(CASE);
////    	return 0;
////    #undef CASE
//        int size = 0;
//        NUMBER_TYPE_CASES(dt, size = sizeof(T));
//        return size;
//    }
    static int v(DataType dt) {
        switch(dt) {
            case DT_FLOAT:
                return sizeof(float);
            case DT_DOUBLE:
                return sizeof(double);
            case DT_INT64:
                return sizeof(int64);
            case DT_INT32:
                return sizeof(int32);
            case DT_INT16:
                return sizeof(int16);
            case DT_INT8:
                return sizeof(int8);
            case DT_UINT64:
                return sizeof(uint64);
            case DT_UINT32:
                return sizeof(uint32);
            case DT_UINT16:
                return sizeof(uint16);
            case DT_UINT8:
                return sizeof(uint8);
            case DT_BOOL:
                return sizeof(bool);
            default:
                LOG(FATAL) << "Invalid type: " << dt;
        }
        return 0;
    }
};

#define TYPE_SWITCH_CASE(TYPE,STMT)					\
	case DataTypeToEnum<TYPE>::value: {				\
		typedef TYPE T;								\
		STMT;										\
		break;										\
}

#define TYPE_CASES(ENUM, STMT, CALL)				\
	switch (ENUM) {									\
		CALL(TYPE_SWITCH_CASE,STMT)					\
	default:										\
		LOG(FATAL) << "Invalid type: " << ENUM;     \
	}

#define NUMBER_TYPE_CASES(ENUM,STMT) TYPE_CASES(ENUM,STMT,CALL_NUMBER_TYPES)

#define NUM_SWITCH_CASE(NESTNUM,CONSTEXPR_NUM,STMT)       \
case CONSTEXPR_NUM: {                                     \
    constexpr int N##NESTNUM = CONSTEXPR_NUM;             \
    STMT;                                                 \
    break;                                                \
}                                                       \

#define NUMBER_CASES_NESTED(NESTNUM,NUM,STMT)           \
switch (NUM) {                                      \
NUM_SWITCH_CASE(NESTNUM,1,STMT)                     \
NUM_SWITCH_CASE(NESTNUM,2,STMT)                     \
NUM_SWITCH_CASE(NESTNUM,3,STMT)                     \
NUM_SWITCH_CASE(NESTNUM,4,STMT)                     \
NUM_SWITCH_CASE(NESTNUM,5,STMT)                     \
NUM_SWITCH_CASE(NESTNUM,6,STMT)                     \
NUM_SWITCH_CASE(NESTNUM,7,STMT)                     \
NUM_SWITCH_CASE(NESTNUM,8,STMT)                     \
NUM_SWITCH_CASE(NESTNUM,9,STMT)                     \
default:                                            \
    LOG(FATAL) << NUM << " is not a hardcoded number type case"; \
}

#define NUMBER_CASES(NUM,STMT) NUMBER_CASES_NESTED(,NUM,STMT)


#endif // TYPES_H
