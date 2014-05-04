/*
 * @file render_manager.cpp 
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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Render/Render_Base/render_base.h"
#include "Physika_Render/Render_Manager/render_manager.h"
using std::list;

namespace Physika{

RenderManager::RenderManager()
{
}

RenderManager::~RenderManager()
{
}

unsigned int RenderManager::numRenderTasks() const
{
    return render_list_.size();
}

void RenderManager::insertBack(RenderBase *render_task)
{
    PHYSIKA_ASSERT(render_task);
    render_list_.push_back(render_task);
}

void RenderManager::insertFront(RenderBase *render_task)
{
    PHYSIKA_ASSERT(render_task);
    render_list_.push_front(render_task);
}

void RenderManager::insertAtIndex(unsigned int index, RenderBase *task)
{
    PHYSIKA_ASSERT(index>=0);
    PHYSIKA_ASSERT(index<render_list_.size());
    PHYSIKA_ASSERT(task);
    list<RenderBase*>::iterator pos = render_list_.begin();
    while(index-- >= 0)
	++pos;
    render_list_.insert(pos,task);
}

void RenderManager::removeBack()
{
    render_list_.pop_back();
}

void RenderManager::removeFront()
{
    render_list_.pop_front();
}

void RenderManager::removeAtIndex(unsigned int index)
{
    PHYSIKA_ASSERT(index>=0);
    PHYSIKA_ASSERT(index<render_list_.size());
    list<RenderBase*>::iterator pos = render_list_.begin();
    while(index-- >= 0)
	++pos;
    render_list_.erase(pos);
}

void RenderManager::removeAll()
{
    render_list_.clear();
}

void RenderManager::renderAll()
{
    for(list<RenderBase*>::iterator iter = render_list_.begin(); iter != render_list_.end(); ++iter)
    {
	RenderBase *cur_render = *iter;
	if(cur_render)
	    cur_render->render();
    }
}

void RenderManager::renderAtIndex(unsigned int index)
{
    PHYSIKA_ASSERT(index>=0);
    PHYSIKA_ASSERT(index<render_list_.size());
    list<RenderBase*>::iterator iter = render_list_.begin();
    while(index-- >= 0)
	++iter;
    RenderBase *cur_render = *iter;
    if(cur_render)
	cur_render->render();
}

}  //end of namespace Physika


















