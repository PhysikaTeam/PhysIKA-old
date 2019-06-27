//#include <boost/type_traits.hpp>

#include <GL/glew.h>
#include <iostream> 
#include <qevent.h>
#include "QLichtWidget.h"
using namespace Physika;

QLichtWidget::QLichtWidget(QWidget* parent) :
	QWidget(parent)
	, m_zoomLimit(1.0f)
	, m_zoomSpeed(1.0f)
	, m_rotationSpeed(-1.0f)
	, m_translateSpeed(1.0f)
{
	m_camera = NULL;
	driver = NULL;
	smgr = NULL;
	guienv = NULL;
	device = NULL;
}

QLichtWidget::~QLichtWidget() {
}


void QLichtWidget::setActiveCameraMaya()
{
	if (0 == device)
		return;

	m_camera = smgr->addCameraSceneNodeMaya(0, 10.0f*m_rotationSpeed, 0.01f*m_zoomSpeed, 10.0f*m_translateSpeed);
	m_camera->setFarValue(100.f);
	m_camera->setTarget(core::vector3df(0, 0, 0));

	m_camera->setInputReceiverEnabled(true);
	device->getSceneManager()->setActiveCamera(m_camera);
}

int marker = 0;
void QLichtWidget::autoUpdateIrrlicht(irr::IrrlichtDevice* device)
{	
	SceneGraph& scenegraph = SceneGraph::getInstance();

	scenegraph.takeOneFrame();


	driver->beginScene(true, true, SColor(255, 100, 101, 140));
 	smgr->drawAll();
// 	driver->draw2DLine(core::vector2d<s32>(10.0, 10.0f), core::vector2d<s32>(50.0, 50.0f), SColor(255, 255, 25, 255));
 	driver->draw3DLine(core::vector3df(10.0, 10.0f, 10.0f), core::vector3df(50.0, 50.0f, 50.0f), SColor(255, 255, 255, 255));

	glLineWidth(4);
	glBegin(GL_LINES);
	glColor3f(1, 0, 0);
	glVertex3f(0, 0, 0);
	glVertex3f(1, 0, 0);
	glColor3f(0, 1, 0);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 1, 0);
	glColor3f(0, 0, 1);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, 1);
	glEnd();

//	scenegraph.draw();

	guienv->drawAll();

	driver->endScene();
}

void QLichtWidget::paintEvent(QPaintEvent* e)
{
	if (device != 0)
	{
		emit updateIrrlicht(device);
	}

	QWidget::paintEvent(e);
}

void QLichtWidget::resizeEvent(QResizeEvent* e)
{
	dimension2d<u32> screensize = dimension2d<u32>(width(), height());
	driver->OnResize(screensize);

	f32 fAspect = (float)width() / (float)height();
	smgr->getActiveCamera()->setAspectRatio(fAspect);

	QWidget::resizeEvent(e);
}

void QLichtWidget::enterEvent(QEvent*) {
}

void QLichtWidget::leaveEvent(QEvent*) {
}

void QLichtWidget::mouseMoveEvent(QMouseEvent* mouseEvent) {

	irr::SEvent irrEvent;

	irrEvent.EventType = irr::EET_MOUSE_INPUT_EVENT;
	irrEvent.MouseInput.Event = EMIE_MOUSE_MOVED;

	irrEvent.MouseInput.X = mouseEvent->x();
	irrEvent.MouseInput.Y = mouseEvent->y();
	irrEvent.MouseInput.Wheel = 0.0f; // Zero is better than undefined

	device->postEventFromUser(irrEvent);
}

void QLichtWidget::mousePressEvent(QMouseEvent* mouseEvent) {
	if (device != 0)
	{
		sendMouseEventToIrrlicht(mouseEvent, true);
	}
	mouseEvent->accept();
}

void QLichtWidget::mouseReleaseEvent(QMouseEvent* mouseEvent) {
	if (device != 0)
	{
		sendMouseEventToIrrlicht(mouseEvent, false);
	}
	mouseEvent->accept();
}


void QLichtWidget::wheelEvent(QWheelEvent *event)
{
	float numberDegrees = event->delta();

	irr::SEvent irrEvent;

	irrEvent.EventType = irr::EET_MOUSE_INPUT_EVENT;
	irrEvent.MouseInput.Event = irr::EMIE_MOUSE_WHEEL;

	irrEvent.MouseInput.X = 0;
	irrEvent.MouseInput.Y = 0;
	irrEvent.MouseInput.Wheel = 1.0; // Zero is better than undefined
	
	const core::list<scene::ISceneNodeAnimator*>& animatorList = m_camera->getAnimators();
	if (!animatorList.empty())
	{
		scene::ISceneNodeAnimatorCameraMaya* mayaAnimator = static_cast<scene::ISceneNodeAnimatorCameraMaya*>(*animatorList.begin());
		f32 d = mayaAnimator->getDistance();
		
		d += numberDegrees * mayaAnimator->getZoomSpeed();
		d = d < m_zoomLimit ? m_zoomLimit : d;

		mayaAnimator->setDistance(d);
	}

	device->postEventFromUser(irrEvent);

	event->accept();
}

void QLichtWidget::dragEnterEvent(QDragEnterEvent* dragEnterEvent) {
}

void QLichtWidget::dropEvent(QDropEvent* dropEvent) {
}

void QLichtWidget::keyPressEvent(QKeyEvent* k) {
}

void QLichtWidget::keyReleaseEvent(QKeyEvent* k) {
}


void QLichtWidget::timerEvent(QTimerEvent* event)
{
	if (device != 0)
	{
		emit updateIrrlicht(device);
	}
	event->accept();
}


void QLichtWidget::sendMouseEventToIrrlicht(QMouseEvent* event, bool pressedDown)
{
	irr::SEvent irrEvent;

	irrEvent.EventType = irr::EET_MOUSE_INPUT_EVENT;

	switch (event->button())
	{
	case Qt::LeftButton:
		irrEvent.MouseInput.Event = pressedDown ? irr::EMIE_LMOUSE_PRESSED_DOWN : irr::EMIE_LMOUSE_LEFT_UP;
		break;

	case Qt::MidButton:
		irrEvent.MouseInput.Event = pressedDown ? irr::EMIE_MMOUSE_PRESSED_DOWN : irr::EMIE_MMOUSE_LEFT_UP;
		break;

	case Qt::RightButton:
		irrEvent.MouseInput.Event = pressedDown ? irr::EMIE_RMOUSE_PRESSED_DOWN : irr::EMIE_RMOUSE_LEFT_UP;
		break;

	default:
		return; // Cannot handle this mouse event
	}

	irrEvent.MouseInput.X = event->x();
	irrEvent.MouseInput.Y = event->y();
	irrEvent.MouseInput.Wheel = 0.0f; // Zero is better than undefined

	device->postEventFromUser(irrEvent);
}


struct SIrrlichtKey
{
	irr::EKEY_CODE code;
	wchar_t ch;
};

SIrrlichtKey convertToIrrlichtKey(int key)
{
	SIrrlichtKey irrKey;
	irrKey.code = (irr::EKEY_CODE)(0);
	irrKey.ch = (wchar_t)(0);

	// Letters A..Z and numbers 0..9 are mapped directly
	if ((key >= Qt::Key_A && key <= Qt::Key_Z) || (key >= Qt::Key_0 && key <= Qt::Key_9))
	{
		irrKey.code = (irr::EKEY_CODE)(key);
		irrKey.ch = (wchar_t)(key);
	}
	else

		// Dang, map keys individually
		switch (key)
		{
		case Qt::Key_Up:
			irrKey.code = irr::KEY_UP;
			break;

		case Qt::Key_Down:
			irrKey.code = irr::KEY_DOWN;
			break;

		case Qt::Key_Left:
			irrKey.code = irr::KEY_LEFT;
			break;

		case Qt::Key_Right:
			irrKey.code = irr::KEY_RIGHT;
			break;
		}
	return irrKey;
}

void QLichtWidget::sendKeyEventToIrrlicht(QKeyEvent* event, bool pressedDown)
{
	irr::SEvent irrEvent;

	irrEvent.EventType = irr::EET_KEY_INPUT_EVENT;

	SIrrlichtKey irrKey = convertToIrrlichtKey(event->key());

	if (irrKey.code == 0) return; // Could not find a match for this key

	irrEvent.KeyInput.Key = irrKey.code;
	irrEvent.KeyInput.Control = ((event->modifiers() & Qt::ControlModifier) != 0);
	irrEvent.KeyInput.Shift = ((event->modifiers() & Qt::ShiftModifier) != 0);
	irrEvent.KeyInput.Char = irrKey.ch;
	irrEvent.KeyInput.PressedDown = pressedDown;

	device->postEventFromUser(irrEvent);
}

void QLichtWidget::initIrrlicht()
{
	setMouseTracking(true);

	irr::SIrrlichtCreationParameters param;
	param.WindowId = (void*)winId();
	param.DriverType = E_DRIVER_TYPE::EDT_OPENGL;
	device = createDeviceEx(param);

	if (!device)
		return;

	device->setWindowCaption(L"Hello World! - Irrlicht Engine Demo");
	device->setResizable(true);

	driver = device->getVideoDriver();
	smgr = device->getSceneManager();
	guienv = device->getGUIEnvironment();

	guienv->addStaticText(L"Hello World! This is the Irrlicht Software renderer!",
		irr::core::rect<s32>(10, 10, 260, 22), true);



	IAnimatedMesh* mesh = smgr->getMesh("../Media/sydney.md2");
	if (!mesh)
	{
		device->drop();
		return;
	}
	IAnimatedMeshSceneNode* node = smgr->addAnimatedMeshSceneNode(mesh);

	if (node)
	{
		node->setMaterialFlag(EMF_LIGHTING, false);
		node->setMD2Animation(scene::EMAT_STAND);
		node->setMaterialTexture(0, driver->getTexture("../Media/sydney.bmp"));
	}

	setActiveCameraMaya();

//	setAttribute(Qt::WA_OpaquePaintEvent);

	connect(this, SIGNAL(updateIrrlicht(irr::IrrlichtDevice*)),
		this, SLOT(autoUpdateIrrlicht(irr::IrrlichtDevice*)));
	startTimer(0);
}
