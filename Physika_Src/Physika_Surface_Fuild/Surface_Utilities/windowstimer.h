#ifndef PHYSIKA_SURFACE_FUILD_SURFACE_UTILITIES_WINDOWSTIMER_H_
#define PHYSIKA_SURFACE_FUILD_SURFACE_UTILITIES_WINDOWSTIMER_H_

#include <Windows.h>

namespace Physika{
class WindowsTimer {
public:
	typedef double time_t;
	WindowsTimer() {
		LARGE_INTEGER large_interger;
		QueryPerformanceFrequency(&large_interger);
		freq = (time_t)large_interger.QuadPart;
		restart();
	}
	void restart() {
		LARGE_INTEGER large_interger;
		QueryPerformanceCounter(&large_interger);
		start = large_interger.QuadPart;
	}
	void record() {
		LARGE_INTEGER large_interger;
		QueryPerformanceCounter(&large_interger);
		end = large_interger.QuadPart;
	}
	inline void stop() {
		record();
	}
	time_t get() const { return (end - start) / freq; }
private:
	__int64 start, end;
	time_t freq;
};
}
#endif
