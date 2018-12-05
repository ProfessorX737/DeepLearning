#include "logging.h"
#include <stdio.h>

LogMessage::LogMessage(Severity severity, const char* fileName, const char* funcName, int line) :
	severity_(severity), fileName_(fileName), funcName_(funcName), line_(line) {
}

LogMessage::~LogMessage() {
	printLogMessage();
}

void LogMessage::printLogMessage() {
	fprintf(stderr, "%s: %s:%s:%d '%s'\n", SeverityNames[severity_], fileName_, funcName_, line_, str().c_str());
}

LogMessageFatal::LogMessageFatal(const char* fileName, const char* funcName, int line) :
	LogMessage(FATAL, fileName, funcName, line) {
}

LogMessageFatal::~LogMessageFatal() {
	printLogMessage();
	abort();
}
