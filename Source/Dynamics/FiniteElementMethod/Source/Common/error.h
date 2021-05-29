#ifndef HJ_ERROR_H_
#define HJ_ERROR_H_

#include <iostream>

template <typename T>
void exit_if(const T&v, const char *msg = 0) {
  if(v){
    printf("%s", msg);
    exit(v);
  } 
}

#define IF_ERR(end_op, ...) {                   \
    const int err(__VA_ARGS__);                 \
    if(err)                                     \
      end_op(err);                              \
  }

#endif
