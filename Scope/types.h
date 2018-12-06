#pragma once

#include <string>
#include <tuple>

typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

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
	}

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

#define CALL_float(m) m(float)
#define CALL_double(m) m(double)
#define CALL_int64(m) m(int64)
#define CALL_int32(m) m(int32)
#define CALL_int16(m) m(int16)
#define CALL_int8(m) m(int8)
#define CALL_uint64(m) m(uint64)
#define CALL_uint32(m) m(uint32)
#define CALL_uint16(m) m(uint16)
#define CALL_uint8(m) m(uint8)
#define CALL_string(m) m(std::string)
#define CALL_bool(m) m(bool)

#define CALL_INTEGRALS(m)			\
CALL_int64(m)						\
CALL_int32(m)						\
CALL_int16(m)						\
CALL_int8(m)						\
CALL_uint64(m)						\
CALL_uint32(m)						\
CALL_uint16(m)						\
CALL_uint8(m)

#define CALL_NUMBER_TYPES(m)		\
CALL_INTEGRALS(m)					\
CALL_float(m)						\
CALL_double(m)

#define CALL_POD_TYPES(m)			\
CALL_NUMBER_TYPES(m)				\
CALL_bool(m)

#define CALL_ALL_TYPES(m)			\
CALL_POD_TYPES(m);					\
CALL_string(m)

int DataTypeSize(DataType dt);
