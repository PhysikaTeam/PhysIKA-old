#ifndef TCDSM_UTIL_EXPORT_H
#define TCDSM_UTIL_EXPORT_H

// 定义导出配置
#define TCDSM_UTIL_LIBRARY

#if defined(_MSC_VER) && defined(TCDSM_UTIL_DISABLE_MSVC_WARNINGS)
    #pragma warning( disable : 4244 )
    #pragma warning( disable : 4251 )
    #pragma warning( disable : 4267 )
    #pragma warning( disable : 4275 )
    #pragma warning( disable : 4290 )
    #pragma warning( disable : 4786 )
    #pragma warning( disable : 4305 )
    #pragma warning( disable : 4996 )
#endif

#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) \
  || defined( __BCPLUSPLUS__) || defined( __MWERKS__)
    #  if defined( TCDSM_UTIL_STATIC )
    #    define TCDSM_UTIL_EXPORT
    #  elif defined( TCDSM_UTIL_LIBRARY )
    #    define TCDSM_UTIL_EXPORT   __declspec(dllexport)
    #  else
    #    define TCDSM_UTIL_EXPORT   __declspec(dllimport)
    #  endif
#else
    #  define TCDSM_UTIL_EXPORT
#endif


#endif // TCDSM_UTIL_EXPORT_H
