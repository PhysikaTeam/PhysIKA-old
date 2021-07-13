#pragma once

#include <core/animation/animation.h>
#include <core/utils/logging.h>
#include <core/utils/timer.h>

namespace pbal {

class PhysicsAnimation : public Animation
{
public:
    PhysicsAnimation()
    {
        _currentFrame.index = 0;
    }

    virtual ~PhysicsAnimation() {}

    void advanceSingleFrame()
    {
        Frame f = _currentFrame;
        update(++f);
    }

    Frame currentFrame() const
    {
        return _currentFrame;
    }

    void setCurrentFrame(const Frame& frame)
    {
        _currentFrame = frame;
    }

    double currentTime() const
    {
        return _currentTime;
    }

protected:
    virtual void onAdvanceTimeStep(double dt) = 0;

    virtual unsigned int numberOfSubTimeSteps(
        double dt) const = 0;

    virtual void onInitialize() {}

private:
    Frame  _currentFrame;
    double _currentTime = 0.0;

    void onUpdate(const Frame& frame) final
    {
        if (frame.index > _currentFrame.index)
        {
            if (_currentFrame.index <= 0)
            {
                initialize();
            }

            int numberOfFrames = frame.index - _currentFrame.index;

            for (int i = 0; i < numberOfFrames; ++i)
            {
                advanceTimeStep(frame.dt);
            }

            _currentFrame = frame;
        }
    }

    void advanceTimeStep(double dt)
    {
        _currentTime = _currentFrame.time();

        // Perform adaptive time-stepping
        double remainingTime = dt;
        while (remainingTime > 0.0)
        {
            unsigned int numSteps = numberOfSubTimeSteps(remainingTime);
            double       actualTimeInterval =
                remainingTime / static_cast<double>(numSteps);

            LOG_INFO << "Number of remaining sub-timesteps: " << numSteps;

            LOG_INFO << "Begin onAdvanceTimeStep: " << actualTimeInterval
                     << " (1/" << 1.0 / actualTimeInterval << ") seconds";

            Timer timer;
            onAdvanceTimeStep(actualTimeInterval);

            LOG_INFO << "End onAdvanceTimeStep (took "
                     << timer.durationInSeconds() << " seconds)";

            remainingTime -= actualTimeInterval;
            _currentTime += actualTimeInterval;
        }
    }

    void initialize()
    {
        onInitialize();
    }
};

}  // namespace pbal
