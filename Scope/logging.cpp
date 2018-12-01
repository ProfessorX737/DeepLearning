#include "logging.h"
#include <stdio.h>

LogMessage::LogMessage(Severity severity)
	: LogMessage(__FILE__, __FUNCTION__, __LINE__, severity) {}

LogMessage::LogMessage(const char* fileName, const char* funcName, int line,
	Severity severity) : 
	fileName_(fileName), funcName_(funcName),
	line_(line), severity_(severity) {}

LogMessage::~LogMessage() {
	printLogMessage();
}
void LogMessage::printLogMessage() {
	fprintf(stderr, "%s: %s:%s:%d '%s'\n", SeverityNames[severity_].c_str(), 
		fileName_, funcName_, line_, str().c_str());
}

LogMessageFatal::LogMessageFatal() : LogMessage(FATAL) {}

LogMessageFatal::LogMessageFatal(const char* fileName, const char* funcName,
	int line) :
	LogMessage(fileName, funcName, line, FATAL) {}

LogMessageFatal::~LogMessageFatal() {
	printLogMessage();
	abort();
}
