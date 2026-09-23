#ifndef CCML_EXPORT_H
#define CCML_EXPORT_H

/* Symbol visibility. CCML_BUILD_LIBRARY is defined only while building ccml
   itself; CCML_STATIC is defined for consumers of a static build. */
#if defined(_WIN32)
#  if defined(CCML_STATIC)
#    define CCML_API
#  elif defined(CCML_BUILD_LIBRARY)
#    define CCML_API __declspec(dllexport)
#  else
#    define CCML_API __declspec(dllimport)
#  endif
#else
#  if defined(CCML_BUILD_LIBRARY)
#    define CCML_API __attribute__((visibility("default")))
#  else
#    define CCML_API
#  endif
#endif

#endif /* CCML_EXPORT_H */
