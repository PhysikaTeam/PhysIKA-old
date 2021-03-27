#ifndef TCDSM_MODELER_EXPORT_H
#define TCDSM_MODELER_EXPORT_H

#define TCDSM_MODELER_LIBRARY

#if defined(_MSC_VER) && defined(TCDSM_MODELER_DISABLE_MSVC_WARNINGS)
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
    #  if defined( TCDSM_MODELER_STATIC )
    #    define TCDSM_MODELER_EXPORT
    #  elif defined( TCDSM_MODELER_LIBRARY )
    #    define TCDSM_MODELER_EXPORT   __declspec(dllexport)
    #  else
    #    define TCDSM_MODELER_EXPORT   __declspec(dllimport)
    #  endif
#else
    #  define TCDSM_MODELER_EXPORT
#endif



#endif // MODELER_GLOBAL_H
