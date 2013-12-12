/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 11 Dec 08:42:53 2013 
 *
 * @brief Some C++ tricks to make our life dealing with Python references a bit
 * easier
 */

#include <Python.h>
#include <memory>

/**
 * Calls Py_DECREF(x) on the input object x. Usage pattern:
 *
 * PyObject* x = ... // builds x with a new python reference
 * auto protected_x = make_safe(x);
 * 
 * After this point, no need to worry about DECREF'ing x anymore. 
 * You can still use `x' inside your code, or protected_x.get().
 */
template <typename T> std::shared_ptr<T> make_safe(T* o) {
  return std::shared_ptr<T>(o, [&](T* p){Py_DECREF(p);});
}

/**
 * Calls Py_XDECREF(x) on the input object x. Usage pattern:
 *
 * PyObject* x = ... // builds x with a new python reference, x may be NULL
 * auto protected_x = make_xsafe(x);
 * 
 * After this point, no need to worry about XDECREF'ing x anymore.
 * You can still use `x' inside your code, or protected_x.get(). Note
 * `x' may be NULL with this method.
 */
template <typename T> std::shared_ptr<T> make_xsafe(T* o) {
  return std::shared_ptr<T>(o, [&](T* p){Py_XDECREF(p);});
}
