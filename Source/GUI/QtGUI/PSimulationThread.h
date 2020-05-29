#ifndef PSIMULATIONTHREAD_H
#define PSIMULATIONTHREAD_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>

namespace PhysIKA
{
	class PSimulationThread : public QThread
	{
		Q_OBJECT

	public:
		static PSimulationThread* instance();
		

		void pause();
		void resume();
		void stop();

		void run() override;

		void startRendering();
		void stopRendering();

		void setTotalFrames(int num);

	Q_SIGNALS:
		//Note: should not be emitted from the user
		void oneFrameFinished();

	private:
		PSimulationThread();

		int max_frames;

		bool m_paused = false;
		bool m_rendering = false;

		QMutex m_mutex;
	};
}


#endif // PSIMULATIONTHREAD_H
