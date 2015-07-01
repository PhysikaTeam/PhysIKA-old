/*
 * @file render_manager.cpp 
 * @Brief maintains a list of render tasks.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
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
    if(render_task)
        render_list_.push_back(render_task);
    else
        std::cerr<<"Cannot insert NULL render task to RenderManager, operation ignored!\n";
}

void RenderManager::insertFront(RenderBase *render_task)
{
    if(render_task)
        render_list_.push_front(render_task);
    else
        std::cerr<<"Cannot insert NULL render task to RenderManager, operation ignored!\n";
}

void RenderManager::insertAtIndex(unsigned int index, RenderBase *task)
{
    bool index_valid = (index<render_list_.size());
    if(!index_valid)
        throw PhysikaException("Render task index out of range!");
    if(task)
    {
        list<RenderBase*>::iterator pos = render_list_.begin();
        while(index != 0)
        {
            --index;
            ++pos;
        }
        render_list_.insert(pos,task);
    }
    else
        std::cerr<<"Cannot insert NULL render task to RenderManager, operation ignored!\n";
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
    bool index_valid = (index<render_list_.size());
    if(!index_valid)
        throw PhysikaException("Render task index out of range!");
    list<RenderBase*>::iterator pos = render_list_.begin();
    while(index != 0)
    {
        --index;
        ++pos;
    }
    render_list_.erase(pos);
}

void RenderManager::removeAll()
{
    render_list_.clear();
}

const RenderBase* RenderManager::taskAtIndex(unsigned int index) const
{
    bool index_valid = (index<render_list_.size());
    if(!index_valid)
        throw PhysikaException("Render task index out of range!");
    list<RenderBase*>::const_iterator iter = render_list_.begin();
    while(index != 0)
    {
        --index;
        ++iter;
    }
    RenderBase *cur_render = *iter;
    return cur_render;
}

RenderBase* RenderManager::taskAtIndex(unsigned int index)
{
    bool index_valid = (index<render_list_.size());
    if(!index_valid)
        throw PhysikaException("Render task index out of range!");
    list<RenderBase*>::iterator iter = render_list_.begin();
    while(index != 0)
    {
        --index;
        ++iter;
    }
    RenderBase *cur_render = *iter;
    return cur_render;
}

int RenderManager::taskIndex(RenderBase *task) const
{
    if(task==NULL)
        return -1;
    list<RenderBase*>::const_iterator iter = render_list_.begin();
    int index = 0;
    while(iter != render_list_.end())
    {
        if(*iter == task)
            return index;
        ++iter;
        ++index;
    }
    return -1;
}

void RenderManager::renderAll()
{
    for(list<RenderBase*>::iterator iter = render_list_.begin(); iter != render_list_.end(); ++iter)
    {
        RenderBase *cur_render = *iter;
        PHYSIKA_ASSERT(cur_render);
        cur_render->render();
    }
}

void RenderManager::renderAtIndex(unsigned int index)
{
    bool index_valid = (index<render_list_.size());
    if(!index_valid)
        throw PhysikaException("Render task index out of range!");
    list<RenderBase*>::iterator iter = render_list_.begin();
    while(index != 0)
    {
        --index;
        ++iter;
    }
    RenderBase *cur_render = *iter;
    PHYSIKA_ASSERT(cur_render);
    cur_render->render();
}

}  //end of namespace Physika
