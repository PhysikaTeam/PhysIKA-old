#include "Timer.h"
using namespace Physika;

CTimer::CTimer()
{
#ifdef _WIN32
	use_qpc = QueryPerformanceFrequency(&t_freq);
	t_init.QuadPart = t_end.QuadPart = 0;
#else
	t_init.tv_sec = t_init.tv_usec = 0;
	t_end.tv_sec = t_end.tv_usec = 0;
#endif
}

CTimer::~CTimer()
{

}

void CTimer::Start()
{
#ifdef _WIN32
	QueryPerformanceCounter(&t_init);
#else
	gettimeofday(&t_init, NULL);
#endif
}

double CTimer::Time()
{
#ifdef _WIN32
	QueryPerformanceCounter(&t_end);
	return (use_qpc ? (t_end.QuadPart - t_init.QuadPart)*1000.0 / t_freq.QuadPart : -1.0); // todo what if qpc is false
#else
	return (t_end.tv_sec - t_init.tv_sec)*1000.0 + (t_end.tv_usec - t_init.tv_usec)/1000.0;
#endif
}
