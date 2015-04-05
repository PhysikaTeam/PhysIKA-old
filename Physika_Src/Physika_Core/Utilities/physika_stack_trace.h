/*
 * @file physika_stack_trace.h 
 * @brief call stack, used by physika exceptions to display trace history
 * @acknowledge       http://stacktrace.sourceforge.net/
 *                    http://stackwalker.codeplex.com/
 * @author FeiZhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_UTILITIES_PHYSIKA_STACK_TRACE_H_
#define PHYSIKA_CORE_UTILITIES_PHYSIKA_STACK_TRACE_H_

#include <string>
#include <vector>

namespace Physika{

class PhysikaStackTrace
{
public:
    PhysikaStackTrace();
    ~PhysikaStackTrace() throw();
    std::string toString() const;  //serialize entire call stack into a string
protected:
    //internal class, entry of call stack
    class StackEntry
    {
    public:
        StackEntry();
        ~StackEntry();
        std::string toString() const;
    public:
        std::string file_name_;
        std::string function_name_;
        unsigned int line_num_; //not implemented for *unix
    };
protected:
    std::vector<StackEntry> stack_;
};
    
}  //end of namespace Physika

#endif //PHYSIKA_CORE_UTILITIES_PHYSIKA_STACK_TRACE_H_
