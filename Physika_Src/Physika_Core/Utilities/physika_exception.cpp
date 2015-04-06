/*
 * @file physika_exception.cpp 
 * @brief Customized exception with stack trace information for Physika, throw an exception 
 *        in case of error so that callers of Physika can handle it.
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

#include "Physika_Core/Utilities/physika_exception.h"

namespace Physika{

PhysikaException::PhysikaException(const std::string &msg) throw()
    :std::exception(),error_msg_(msg)     
#ifdef PHYSIKA_EXCEPTION_WITH_STACK_TRACE
    ,stack_trace_()
#endif
{
}
 
PhysikaException::~PhysikaException() throw()
{
}

const char* PhysikaException::what() const throw()
{
#ifdef PHYSIKA_EXCEPTION_WITH_STACK_TRACE
    std::string msg = std::string("[") + error_msg_ + std::string("]\n");
    msg = msg + stack_trace_.toString();
    return msg.c_str();    
#else
    return error_msg_.c_str();
#endif
}
  
std::string PhysikaException::errorMessage() const throw()
{
#ifdef PHYSIKA_EXCEPTION_WITH_STACK_TRACE
    std::string msg = std::string("[") + error_msg_ + std::string("]\n");
    msg = msg + stack_trace_.toString();
    return msg;    
#else
    return error_msg_;
#endif
}
  
} //end of namespace Physika
    
