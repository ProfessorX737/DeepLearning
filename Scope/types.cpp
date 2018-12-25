#include "types.h"

int DataTypeSize(DataType dt) {
#define CASE(T,...) if(DataTypeToEnum<T>::v() == dt) { return sizeof(T); }
	CALL_POD_TYPES(CASE);
	return 0;
#undef CASE
}
