#pragma once

#include "Initilizer.h"

template<typename T = float>
class ZeroInitializer : public Initializer<T> {
private:
	T * m_data;
	T m_mean;
	T m_stddev;
public:
	ZeroInitializer() {}
	~ZeroInitializer() { delete m_data; }
	T* data(int size);
};

template<typename T>
T* ZeroInitializer<T>::data(int size) {
	m_data = new T[size]();
	return m_data;
}
