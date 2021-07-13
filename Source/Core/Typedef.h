#pragma once
#include <list>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <algorithm>

using uint   = unsigned int;
using String = std::string;

template <class T>
using VectorPtr = std::vector<std::shared_ptr<T>>;

template <class T>
using List = std::list<T>;

template <class T>
using ListPtr = std::list<std::shared_ptr<T>>;

template <class T>
using Map = std::map<std::string, T>;

template <class T>
using MapPtr = std::map<std::string, std::shared_ptr<T>>;

template <class T>
using MultiMap = std::multimap<std::string, T>;

template <class T>
using MultiMapPtr = std::multimap<std::string, std::shared_ptr<T>>;

namespace TypeInfo {
template <class T, class... Args>
std::shared_ptr<T> New(Args&&... args)
{
    std::shared_ptr<T> p(new T(std::forward<Args>(args)...));
    return p;
}

template <class TA, class TB>
inline TA* cast(TB* b)
{
    TA* ptr = dynamic_cast<TA*>(b);
    return ptr;
}

template <class TA, class TB>
inline std::shared_ptr<TA> cast(std::shared_ptr<TB> b)
{
    std::shared_ptr<TA> ptr = std::dynamic_pointer_cast<TA>(b);
    return ptr;
}

template <class Derived, class Base>
Derived* CastPointerDown(Base* b)
{
    Derived* ptr = dynamic_cast<Derived*>(b);
    return ptr;
}

template <class Base, class Derived>
Base* CastPointerUp(Derived* b)
{
    Base* ptr = dynamic_cast<Base*>(b);
    return ptr;
}

template <class Derived, class Base>
std::shared_ptr<Derived> CastPointerDown(std::shared_ptr<Base> b)
{
    std::shared_ptr<Derived> ptr = std::dynamic_pointer_cast<Derived>(b);
    return ptr;
}

template <class Base, class Derived>
std::shared_ptr<Base> CastPointerUp(std::shared_ptr<Derived> b)
{
    std::shared_ptr<Base> ptr = std::dynamic_pointer_cast<Base>(b);
    return ptr;
}

}  // namespace TypeInfo
