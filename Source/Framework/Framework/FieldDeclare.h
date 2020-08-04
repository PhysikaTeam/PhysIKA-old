#pragma once
//----------------------------------------------------------------------------
// Check for unsupported old compilers.
#if defined(_MSC_VER) && _MSC_VER < 1800
# error PhysIKA requires MSVC++ 12.0 aka Visual Studio 2013 or newer
#endif

#if !defined(__clang__) && defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8))
# error PhysIKA requires GCC 4.8 or newer
#endif

#include "FieldVar.h"

namespace PhysIKA
{

#define DEF_VAR(name, T, value, type) \
VarField<T> m_##name = VarField<T>(T(value), std::string(#name), "", type, this);			\
VarField<T>& get##name() {return m_##name;}

#define DEF_VAR(name, T, value, type, desc) \
VarField<T> m_##name = VarField<T>(T(value), std::string(#name), desc, type, this);			\
VarField<T>& get##name() {return m_##name;}


#define DEF_ARRAY(name, T, device, type) \
ArrayField<T, device, type> name;	\
ArrayField<T, device, type>& get##name() {return name;}
}    


