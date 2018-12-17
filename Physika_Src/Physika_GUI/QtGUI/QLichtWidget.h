#ifndef G3D_WIDGET_HPP
#define G3D_WIDGET_HPP

#include <memory>
#include <QtWidgets/QOpenGLWidget>
#include <irrlicht.h>

#include <cuda_runtime_api.h>
#include "Physika_Framework/Framework/SceneGraph.h"
#include "Physika_Dynamics/ParticleSystem/ParticleFluid.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Framework/Log.h"
#include "Physika_Render/PointRenderModule.h"

using namespace irr;
using namespace core;
using namespace scene;
using namespace video;
using namespace io;
using namespace gui;

class QLichtWidget : public QWidget
{
    Q_OBJECT

public:
    QLichtWidget(QWidget* parent = NULL);

    virtual ~QLichtWidget();

	void initIrrlicht();
	void setActiveCameraMaya();

	void setZoomLimit(f32 limit) { m_zoomLimit = limit; }
	f32  getZoomLimit() { return m_zoomLimit; }

	void setZoomSpeed(f32 speed) { m_zoomSpeed = speed; }
	f32  getZoomSpeed() { return m_zoomSpeed; }

	void setRotationSpeed(f32 speed) { m_rotationSpeed = speed; }
	f32  getRotationSpeed() { return m_rotationSpeed; }

	void setTranslationSpeed(f32 speed) { m_translateSpeed = speed; }
	f32  getTranslationSpeed() { return m_translateSpeed; }

signals:
	void updateIrrlicht(irr::IrrlichtDevice* device);

public slots:
	void autoUpdateIrrlicht(irr::IrrlichtDevice* device);

protected:
    void paintEvent(QPaintEvent* e) Q_DECL_OVERRIDE;
    void resizeEvent(QResizeEvent* e) Q_DECL_OVERRIDE;
    void enterEvent(QEvent*) Q_DECL_OVERRIDE;
    void leaveEvent(QEvent*) Q_DECL_OVERRIDE;
	void mousePressEvent(QMouseEvent* e) Q_DECL_OVERRIDE;
    void mouseReleaseEvent(QMouseEvent* e) Q_DECL_OVERRIDE;
	void wheelEvent(QWheelEvent *event) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QMouseEvent* e) Q_DECL_OVERRIDE;
    void dragEnterEvent(QDragEnterEvent* e) Q_DECL_OVERRIDE;
    void dropEvent(QDropEvent* e) Q_DECL_OVERRIDE;
    void keyPressEvent(QKeyEvent* e) Q_DECL_OVERRIDE;
    void keyReleaseEvent(QKeyEvent* e) Q_DECL_OVERRIDE;

	virtual void timerEvent(QTimerEvent* event);

	void sendMouseEventToIrrlicht(QMouseEvent* event, bool pressedDown);
	void sendKeyEventToIrrlicht(QKeyEvent* event, bool pressedDown);

private:
	f32 m_zoomSpeed;
	f32 m_rotationSpeed;
	f32 m_translateSpeed;

	f32 m_zoomLimit;

	ICameraSceneNode* m_camera;

	IVideoDriver* driver;
	ISceneManager* smgr;
	IGUIEnvironment* guienv;
	IrrlichtDevice* device;
};

#endif
