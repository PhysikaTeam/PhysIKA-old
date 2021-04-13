#include "Framework/Framework/ModuleVisual.h"
#include "Framework/Framework/Node.h"

namespace PhysIKA
{
VisualModule::VisualModule()
	: Module()
	, m_scale(1.0f)
	, m_translation(0.0f)
	, m_rotation(0.0f, 0.0f, 0.0f, 1.0f)
{
	attachField(&m_visible, "visible", "this is a variable indicating whether this module will be rendered on screen!", false);
	m_visible.setValue(true);
}

VisualModule::~VisualModule()
{
}

void VisualModule::setVisible(bool bVisible)
{
	m_visible.setValue(bVisible);
}

void VisualModule::rotate(float angle, float x, float y, float z)
{
	m_rotation += Quaternion<float>(angle, x, y, z);
}

void VisualModule::translate(float x, float y, float z)
{
	m_translation += Vector3f(x, y, z);
}

void VisualModule::scale(float x, float y, float z)
{
	m_scale += Vector3f(x, y, z);
}

}