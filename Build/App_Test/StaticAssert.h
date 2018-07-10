#ifndef MFD_STATICASSERT_H
#define MFD_STATICASSERT_H

namespace mfd
{
#define JOIN( X, Y ) DO_JOIN( X, Y )
#define DO_JOIN( X, Y ) DO_JOIN2(X,Y)
#define DO_JOIN2( X, Y ) X##Y

	template <bool x> struct STATIC_ASSERTION_FAILURE;

	template <> struct STATIC_ASSERTION_FAILURE<true> { enum { value = 1 }; };

	template<int x> struct static_assert_test{};

#define STATIC_ASSERT( x ) \
	typedef static_assert_test<\
	sizeof(STATIC_ASSERTION_FAILURE< (bool)( x ) >)>\
	JOIN(_static_assert_typedef, __LINE__)

}

#endif