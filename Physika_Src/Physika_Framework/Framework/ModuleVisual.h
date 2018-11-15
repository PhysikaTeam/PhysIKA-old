#pragma once
#include "Framework/Module.h"
#include "Physika_Core/Vectors/vector.h"
#include "Physika_Core/Quaternion/quaternion.h"

namespace Physika
{

class VisualModule : public Module
{
public:
	VisualModule();
	virtual ~VisualModule();

	virtual void display() {};

	void setVisible(bool bVisible);
	bool isVisible() { return m_visible; }

	void rotate(float angle, float x, float y, float z);
	void translate(float x, float y, float z);
	void scale(float x, float y, float z);

	virtual void updateRenderingContext() = 0;

	std::string getModuleType() override { return "VisualModule"; }
protected:
	bool m_visible;

	Quaternion<float> m_rotation;
	Vector3f m_scale;
	Vector3f m_translation;
};

}
