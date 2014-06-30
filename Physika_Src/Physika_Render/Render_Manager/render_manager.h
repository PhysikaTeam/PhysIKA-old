/*
 * @file render_manager.h 
 * @Brief maintains a list of render tasks.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_RENDER_RENDER_MANAGER_RENDER_MANAGER_H_
#define PHYSIKA_RENDER_RENDER_MANAGER_RENDER_MANAGER_H_

#include <list>

namespace Physika{

class RenderBase;

class RenderManager
{
public:
    RenderManager();
    ~RenderManager();
    unsigned int numRenderTasks() const;  //length of the render queue
    void insertBack(RenderBase*);  //insert new task at back of render queue
    void insertFront(RenderBase*); //insert new task at front of render queue
    void insertAtIndex(unsigned int index,RenderBase *task);  //insert new task before the index-th task
    void removeBack(); //remove task at back of render queue
    void removeFront();  //remove task at front of render queue
    void removeAtIndex(unsigned int index);  //remove the index-th task in queue
    void removeAll();  //remove all render tasks  

    void renderAll();
    void renderAtIndex(unsigned int index); //render the index-th task in queue
protected:
    std::list<RenderBase*> render_list_;
};

}  //end of namespace Physika

#endif //PHYSIKA_RENDER_RENDER_MANAGER_RENDER_MANAGER_H_
