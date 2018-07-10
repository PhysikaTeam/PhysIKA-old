#ifndef MFD_IWINDOW_H
#define MFD_IWINDOW_H

#include <vector>
#include <algorithm>
#include <iostream>

#include "IRenderer.h"
#include "Framework/SceneGraph.h"
#include "Camera.h"

using namespace Physika;

class IRenderer;
class IWindow {
public:
	IWindow(void){ renderID = 0; }
	~IWindow(){};

	void SetSimulation(SceneGraph* sim) { _sim = sim; }

	void AddRenderer(IRenderer* iren) { _renders.push_back(iren); }

	void RemoveRenderer(IRenderer* iren) { 
		std::vector<IRenderer*>::iterator iter = find(_renders.begin(), _renders.end(), iren);
		if ( iter != _renders.end() )
			_renders.erase(iter);
	}

	void ShiftRenderingMode()
	{
		renderID++;
		renderID %= (_renders.size()+1);
	}

	void DrawScene(){
//		cout << "render id: " << renderID << endl;
//		assert(_renders.size() > 0);
		if (_renders.size() == 0)
			return;
		
		if (renderID != _renders.size())
		{
			_renders[renderID]->Render(_camera);
		}
		else
		{
			for (int i = 0; i < _renders.size(); i++)
			{
				_renders[i]->Render(_camera);
			}
		}
		
			
	}

	void    SetWindowID(int newWindowID) { windowID = newWindowID; } 
	int     GetWindowID(void) { return( windowID ); }
		
protected:
	int	windowID;

	Camera _camera;
	std::vector<IRenderer*> _renders;
	SceneGraph* _sim;

	int renderID;
};

#endif

