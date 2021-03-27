#include "TimeManager.h"

using namespace SPH;

TimeManager* TimeManager::current = 0;

TimeManager::TimeManager () 
{
	time = 0;
	h = 0.001;
}

TimeManager::~TimeManager () 
{
	current = 0;
}

TimeManager* TimeManager::getCurrent ()
{
	if (current == 0)
	{
		current = new TimeManager ();
	}
	return current;
}

void TimeManager::setCurrent (TimeManager* tm)
{
	current = tm;
}

bool TimeManager::hasCurrent()
{
	return (current != 0);
}

Real TimeManager::getTime()
{
	return time;
}

void TimeManager::setTime(Real t)
{
	time = t;
}

Real TimeManager::getTimeStepSize()
{
	return h;
}

void TimeManager::setTimeStepSize(Real tss)
{
	h = tss;
}

void SPH::TimeManager::saveState(BinaryFileWriter &binWriter)
{
	binWriter.write(time);
	binWriter.write(h);
}

void SPH::TimeManager::loadState(BinaryFileReader &binReader)
{
	binReader.read(time);
	binReader.read(h);
}
