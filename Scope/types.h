#pragma once

typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

//https://github.com/jackpaparian/tensorflow-package/blob/master/tensorflow/include/tensorflow/core/framework/types.pb.h
enum DataType {
	DT_FLOAT = 1,
	DT_DOUBLE = 2,
	DT_INT32 = 3,
	DT_INT16 = 4,
	DT_INT8 = 5,
	DT_UINT32 = 6,
	DT_UINT16 = 7,
	DT_UINT8 = 8,
	DT_STRING = 9,
	DT_BOOL = 10
};

//https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.h

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
	};											\
	template<>									\
	struct EnumToDataType<ENUM> {				\
		static constexpr bool valid = true;		\
		typedef TYPE type;						\
	};											\

TYPE_AND_ENUM(float, DT_FLOAT);
TYPE_AND_ENUM(double, DT_DOUBLE);
TYPE_AND_ENUM(int32, DT_INT32);
TYPE_AND_ENUM(int16, DT_INT16);
TYPE_AND_ENUM(int8, DT_INT8);
TYPE_AND_ENUM(uint32, DT_UINT32);
TYPE_AND_ENUM(uint16, DT_UINT16);
TYPE_AND_ENUM(uint8, DT_UINT8);
TYPE_AND_ENUM(std::string, DT_STRING);
TYPE_AND_ENUM(bool, DT_BOOL);

#undef TYPE_AND_ENUM