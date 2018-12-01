#pragma once
//https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/default/logging.h

#include <string>
#include <sstream>

#undef ERROR

enum Severity {
	INFO = 0,
	WARNING = 1,
	ERROR = 2,
	FATAL = 3,
	NUM_SEVERITIES = 4 // store the number of severity levels within the enum
};

const std::string SeverityNames[NUM_SEVERITIES] = 
	{ "Info","Warning","ERROR","FATAL" };
	
class LogMessage : public std::ostringstream {
public:	
	LogMessage(Severity severity);
	LogMessage(const char* fileName, const char* funcName, int line, 
		Severity severity);
	~LogMessage();
protected:
	void printLogMessage();
private:
	const char* fileName_;
	const char* funcName_;
	int line_;
	Severity severity_;
};

class LogMessageFatal : public LogMessage {
public:
	LogMessageFatal();
	LogMessageFatal(const char* fileName, const char* funcName, int line);
	~LogMessageFatal();
};

#define LOG_INFO LogMessage(INFO)
#define LOG_WARNING LogMessage(WARNING)
#define LOG_ERROR LogMessage(ERROR)
#define LOG_FATAL LogMessageFatal() 
#define LOG(severity) LOG_##severity

#define CHECK_OP(val1, op, val2)										\
	if (!(val1 op val2))												\
		LogMessageFatal()												\
		<<"Check failed: "<< #val1 <<" "<< #op <<" "<< #val2 <<" "

#define CHECK_EQ(val1, val2) CHECK_OP(val1, ==, val2)
#define CHECK_NE(val1, val2) CHECK_OP(val1, !=, val2)
#define CHECK_LT(val1, val2) CHECK_OP(val1, <, val2)
#define CHECK_LE(val1, val2) CHECK_OP(val1, <=, val2)
#define CHECK_GT(val1, val2) CHECK_OP(val1, >, val2)
#define CHECK_GE(val1, val2) CHECK_OP(val1, >=, val2)
#define CHECK_NOT_NULL(val)  CHECK_NQ(val, nullptr)

#ifndef NDEBUG
	
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2) 
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2) 
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2) 
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2) 
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2) 
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2) 
#define DCHECK_NOT_NULL(val)  CHECK_NE(val, nullptr)

#else

#define DCHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2)
#define DCHECK_LT(val1, val2)
#define DCHECK_LE(val1, val2)
#define DCHECK_GT(val1, val2)
#define DCHECK_GE(val1, val2)
#define DCHECK_NOT_NULL(val)

#endif