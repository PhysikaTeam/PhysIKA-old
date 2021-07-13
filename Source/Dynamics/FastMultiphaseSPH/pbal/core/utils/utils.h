#pragma once

#include <algorithm>
#include <iterator>
#include <vector>

namespace pbal {

// Source code from:
// http://en.cppreference.com/w/cpp/algorithm/lower_bound
template <class ForwardIt, class T, class Compare = std::less<>>
ForwardIt binaryFind(ForwardIt first, ForwardIt last, const T& value, Compare comp = {})
{
    // Note: BOTH type T and the type after ForwardIt is dereferenced
    // must be implicitly convertible to BOTH Type1 and Type2, used in Compare.
    // This is stricter than lower_bound requirement (see above)

    first = std::lower_bound(first, last, value, comp);
    return first != last && !comp(value, *first) ? first : last;
}

}  // namespace pbal