#ifndef ISPH_TIMER_H
#define ISPH_TIMER_H

#ifdef _WIN32
	#include <windows.h>
#else
	#include <cstdio>
	#include <sys/time.h>
#endif

namespace Physika {

	/*!
	 *	\class	Timer
	 *	\brief	Measure time intervals in milliseconds.
	 */
	class Timer
	{
	public:

		Timer();
		virtual ~Timer();

		/*!
		 *	\brief	Start the timing.
		 */
		void Start();

		/*!
		 *	\brief	Get the time elapsed, in milliseconds.
		 */
		double Time();

	protected:

#ifdef _WIN32
		BOOL use_qpc;
		LARGE_INTEGER t_freq, t_init, t_end;
#else
		timeval t_init, t_end;
#endif

	};


} // namespace isph

#endif
