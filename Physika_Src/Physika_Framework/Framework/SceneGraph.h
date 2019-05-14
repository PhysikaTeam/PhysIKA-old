#pragma once
#include "Physika_Framework/Framework/Base.h"
#include "Physika_Framework/Framework/Node.h"

namespace Physika {
class SceneGraph : public Base
{
public:
	void setRootNode(std::shared_ptr<Node> root) { m_root = root; }
	std::shared_ptr<Node> getRootNode() { return m_root; }

	virtual bool initialize();
	bool isInitialized() { return m_initialized; }

	virtual void draw();
	virtual void advance(float dt);
	virtual void takeOneFrame();
	virtual void run();

	virtual bool load(std::string name);

	virtual void invoke(unsigned char type, unsigned char key, int x, int y) {};

	template<class TNode, class ...Args>
	std::shared_ptr<TNode> createNewScene(Args&& ... args)
	{
		std::shared_ptr<TNode> root = TypeInfo::New<TNode>(std::forward<Args>(args)...);
		m_root = TypeInfo::CastPointerUp<Node>(root);
		return root;
	}

public:
	static SceneGraph& getInstance();

	inline void setTotalTime(float t) { m_maxTime = t; }
	inline float getTotalTime() { return m_maxTime; }

	inline void setFrameRate(float frameRate) { m_frameRate = frameRate; }
	inline float getFrameRate() { return m_frameRate; }
	inline float getTimeCostPerFrame() { return m_frameCost; }
	inline float getFrameInterval() { return 1.0f / m_frameRate; }
	inline int getFrameNumber() { return m_frameNumber; }

	void setGravity(float g);
	float getGravity();

private:
	SceneGraph()
		: m_elapsedTime(0)
		, m_maxTime(0)
		, m_frameRate(25)
		, m_frameNumber(0)
		, m_frameCost(0)
		, m_initialized(false)
	{};

	/**
	* To avoid erroneous operations
	*/
	SceneGraph(const SceneGraph&) {};
	SceneGraph& operator=(const SceneGraph&) {};

	~SceneGraph() {};

private:
	bool m_initialized;

	float m_elapsedTime;
	float m_maxTime;
	float m_frameRate;
	float m_frameCost;

	int m_frameNumber;

	float m_gravity = -9.8f;

private:
	std::shared_ptr<Node> m_root;
};

}
