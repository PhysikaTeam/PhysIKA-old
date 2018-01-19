/*
 * @file render_task_manager.h 
 * @Basic render task manager
 * @author Wei Chen
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

#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Render/Render_Task_Base/render_task_base.h"

#include "render_task_manager.h"

namespace Physika{

unsigned int RenderTaskManager::numRenderTasks() const
{
    return render_task_list_.size();
}

void RenderTaskManager::insertBack(std::shared_ptr<RenderTaskBase> render_task)
{
    if (render_task)
        render_task_list_.push_back(std::move(render_task));
    else
        std::cerr << "error: can not insert NULL render task to RenderTaskManager, operation ignored!\n";
}

void RenderTaskManager::insertFront(std::shared_ptr<RenderTaskBase> render_task)
{
    if (render_task)
        render_task_list_.push_front(std::move(render_task));
    else
        std::cerr << "error: can not insert NULL render task to RenderTaskManager, operation ignored!\n";
}

void RenderTaskManager::insertAtIndex(unsigned int index, std::shared_ptr<RenderTaskBase> render_task)
{
    bool index_valid = (index < render_task_list_.size());
    if (!index_valid)
        throw PhysikaException("error: render task index out of range!");

    if (render_task)
    {
        auto pos = render_task_list_.begin();
        while (index != 0)
        {
            --index;
            ++pos;
        }
        render_task_list_.insert(pos, std::move(render_task));
    }
    else
        std::cerr << "error: can not insert NULL render task to RenderTaskManager, operation ignored!\n";
}

void RenderTaskManager::removeBack()
{
    render_task_list_.pop_back();
}

void RenderTaskManager::removeFront()
{
    render_task_list_.pop_front();
}

void RenderTaskManager::removeAtIndex(unsigned int index)
{
    bool index_valid = (index < render_task_list_.size());
    if (!index_valid)
        throw PhysikaException("error: render task index out of range!");

    auto pos = render_task_list_.begin();
    while (index != 0)
    {
        --index;
        ++pos;
    }
    render_task_list_.erase(pos);
}

void RenderTaskManager::removeAll()
{
    render_task_list_.clear();
}

std::shared_ptr<const RenderTaskBase> RenderTaskManager::taskAtIndex(unsigned int index) const
{
    bool index_valid = (index < render_task_list_.size());
    if (!index_valid)
        throw PhysikaException("error: render task index out of range!");

    auto iter = render_task_list_.begin();
    while (index != 0)
    {
        --index;
        ++iter;
    }
    return *iter;
}

std::shared_ptr<RenderTaskBase> RenderTaskManager::taskAtIndex(unsigned int index)
{
    bool index_valid = (index < render_task_list_.size());
    if (!index_valid)
        throw PhysikaException("error: render task index out of range!");

    auto iter = render_task_list_.begin();
    while (index != 0)
    {
        --index;
        ++iter;
    }
    return *iter;
}

int RenderTaskManager::taskIndex(const std::shared_ptr<const RenderTaskBase> & render_task) const
{
    if (render_task == nullptr)
        return -1;

    int index = 0;
    auto iter = render_task_list_.begin();
    while (iter != render_task_list_.end())
    {
        if (*iter == render_task)
            return index;
        ++iter;
        ++index;
    }
    return -1;
}

void RenderTaskManager::renderAllTasks()
{
    for (auto iter = render_task_list_.begin(); iter != render_task_list_.end(); ++iter)
    {
        std::shared_ptr<RenderTaskBase> & cur_task = *iter;
        cur_task->renderTask();
    }
}

void RenderTaskManager::renderTaskAtIndex(unsigned int index)
{
    bool index_valid = (index < render_task_list_.size());
    if (!index_valid)
        throw PhysikaException("error: render task index out of range!");

    auto iter = render_task_list_.begin();
    while (index != 0)
    {
        --index;
        ++iter;
    }
    std::shared_ptr<RenderTaskBase> & cur_task = *iter;
    cur_task->renderTask();
}

}//end of namespace Physika