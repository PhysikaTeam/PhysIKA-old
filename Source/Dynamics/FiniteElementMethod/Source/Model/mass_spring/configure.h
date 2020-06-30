#ifndef CONFIG_H
#define CONFIG_H

#define ASSERT(x)                                          \
    do {                                                   \
      if (!(x)) {                                          \
        std::cerr << "# error: assertion failed at\n";     \
        std::cerr << __FILE__ << " " << __LINE__ << "\n";  \
        exit(0);                                           \
      }                                                    \
    } while(0);

#define CALL_SUB_PROG(prog)                       \
    int prog(ptree &pt);                          \
    if ( pt.get<string>("prog.value") == #prog )  \
        return prog(pt);

#define RETURN_WITH_COND_TRUE(expr) \
  if ( expr ) return 1;

#endif
