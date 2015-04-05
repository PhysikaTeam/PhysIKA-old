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

//use platform dependent implementations for GNU G++ and Windows
#if defined(__GNUC__)
#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <cstdlib>
#elif defined(_MSC_VER)
#include "Physika_Dependency/StackWalker/StackWalker.h"
#endif

#include <sstream>
#include "Physika_Core/Utilities/physika_stack_trace.h"

namespace Physika{

//implementations of StackEntry
PhysikaStackTrace::StackEntry::StackEntry():line_num_(0)
{
}
    
PhysikaStackTrace::StackEntry::~StackEntry()
{
}

std::string PhysikaStackTrace::StackEntry::toString() const
{
    std::stringstream adaptor;
    adaptor<<file_name_<<" ("<<line_num_<<"): "<<function_name_;
    return adaptor.str();
}

//for windows and msvc, derive a class from StackWalker for customization
#if defined (_MSC_VER)
class StackWalkerAdaptor: public StackWalker
{
public:
    StackWalkerAdaptor():StackWalker(StackWalker::RetrieveVerbose | StackWalker::SymBuildPath){}
    virtual ~StackWalkerAdaptor(){}
protected:
    virtual void OnCallstackEntry(CallstackEntryType eType, CallstackEntry &entry)
    {
        stack_.push_back(entry);
    }
    //discarded methods
    virtual void OnOutput(LPCSTR){}
    virtual void OnSymInit(LPCSTR, DWORD, LPCSTR){ }
    virtual void OnLoadModule(LPCSTR, LPCSTR, DWORD64, DWORD, DWORD, LPCSTR, LPCSTR, ULONGLONG){}
    virtual void OnDbgHelpErr(LPCSTR, DWORD, DWORD64){}
protected:
    friend class PhysikaStackTrace;
    std::vector<CallstackEntry> stack_;
};
#endif
    
//implementations of PhysikaStackTrace
PhysikaStackTrace::PhysikaStackTrace()
{
#if defined(__GNUC__)
    const unsigned int max_depth = 32; //maximum stack depth
    void *trace[max_depth];
    int stack_depth = backtrace(trace,max_depth);
    //skip the first name (current function)
    for(unsigned int i = 1; i < stack_depth; ++i) 
    {
        Dl_info dlinfo;
        if(!dladdr(trace[i], &dlinfo))
            break;
        const char *symname = dlinfo.dli_sname;
        int status;
        char *demangled = abi::__cxa_demangle(symname,NULL,0,&status);
        if(status == 0 && demangled)
            symname = demangled;
        if(dlinfo.dli_fname && symname)
        {
            StackEntry entry;
            entry.file_name_ = dlinfo.dli_fname;
            entry.function_name_ = symname;
            entry.line_num_ = 0;
            stack_.push_back(entry);
        }
        else
            break;
        if(demangled)
            free(demangled);
    }
#elif defined (_MSC_VER)
    StackWalkerAdaptor sw;
    sw.ShowCallstack();
    for(unsigned int i = 0; i < sw.stack_.size(); ++i)
    {
        StackEntry entry;
        entry.file_name_ = sw.stack_[i].lineFileName;
        entry.function_name_ = sw.stack_[i].name;
        entry.line_num_ = sw.stack_[i].lineNumber;
        stack_.push_back(entry);
    }
#endif
}

PhysikaStackTrace::~PhysikaStackTrace() throw()
{
}

std::string PhysikaStackTrace::toString() const
{
    std::stringstream adaptor;
    adaptor<<std::string("Call stack:\n");
    for(unsigned int i = 0; i < stack_.size(); ++i)
        adaptor<<stack_[i].toString()<<"\n";
    return adaptor.str();
}
    
} //end of namespace Physika
