/*
 * @file physika_exception.h 
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

#ifndef PHYSIKA_CORE_UTILITIES_PHYSIKA_EXCEPTION_H_
#define PHYSIKA_CORE_UTILITIES_PHYSIKA_EXCEPTION_H_

#include <string>
#include <exception>
#include "Physika_Core/Utilities/global_config.h"
#ifdef PHYSIKA_EXCEPTION_WITH_STACK_TRACE
#include "Physika_Core/Utilities/physika_stack_trace.h"
#endif

namespace Physika{

class PhysikaException: public std::exception
{
public:
    explicit PhysikaException(const std::string &msg) throw();
    virtual ~PhysikaException() throw();
    virtual const char* what() const throw();
protected:
    PhysikaException(); //prohibit default constructor
protected:
    std::string error_msg_;
#ifdef PHYSIKA_EXCEPTION_WITH_STACK_TRACE
    PhysikaStackTrace stack_trace_;
#endif
};

} //end of namespace Physika

#endif //PHYSIKA_CORE_UTILITIES_PHYSIKA_EXCEPTION_H_
