#pragma once

template<typename T>
class Initializer {
public:
	virtual T* data(int size) = 0;
};
