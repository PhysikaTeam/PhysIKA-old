#ifndef PSIMULATIONTHREAD_H
#define PSIMULATIONTHREAD_H

#include <QThread>

namespace PhysIKA
{
	class PSimulationThread : public QThread
	{
		Q_OBJECT

	public:
		PSimulationThread();

		void run() override;
	};
}


#endif // PSIMULATIONTHREAD_H
