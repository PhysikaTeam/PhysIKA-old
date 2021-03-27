#ifndef __Version_h__
#define __Version_h__

#define STRINGIZE_HELPER(x) #x
#define STRINGIZE(x) STRINGIZE_HELPER(x)
#define WARNING(desc) message(__FILE__ "(" STRINGIZE(__LINE__) ") : Warning: " #desc)

#define GIT_SHA1 "GITDIR-NOTFOUND"
#define GIT_REFSPEC "GITDIR-NOTFOUND"
#define GIT_LOCAL_STATUS "HEAD-HASH-NOTFOUND"

#define SPLISHSPLASH_VERSION "2.9.0"

#ifdef DL_OUTPUT

#endif

#endif
