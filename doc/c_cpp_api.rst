.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue 15 Oct 14:59:05 2013

=========
 C++ API
=========

The C++ API of ``xbob.io`` allows users to leverage from automatic converters
for classes in :py:class:`xbob.io`.  To use the C API, clients should first,
include the header file ``<xbob.io/api.h>`` on their compilation units and
then, make sure to call once ``import_xbob_io()`` at their module
instantiation, as explained at the `Python manual
<http://docs.python.org/2/extending/extending.html#using-capsules>`_.

Here is a dummy C example showing how to include the header and where to call
the import function:

.. code-block:: c++

   #include <xbob.io/api.h>

   PyMODINIT_FUNC initclient(void) {

     PyObject* m Py_InitModule("client", ClientMethods);

     if (!m) return;

     // imports the NumPy C-API 
     import_array();

     // imports blitz.array C-API
     import_blitz_array();

     // imports xbob.core.random C-API
     import_xbob_io();

   }

.. note::

  The include directory can be discovered using
  :py:func:`xbob.io.get_include`.

Generic Functions
-----------------

.. cpp:function:: int PyBobIo_AsTypenum(bob::core::array::ElementType et)

   Converts the input Bob element type into a ``NPY_<TYPE>`` enumeration value.
   Returns ``NPY_NOTYPE`` in case of problems, and sets a
   :py:class:`RuntimeError`.

.. cpp:function:: PyObject* PyBobIo_TypeInfoAsTuple (const bob::core::array::typeinfo& ti)

   Converts the ``bob::core::array::typeinfo&`` object into a **new reference**
   to a :py:class:`tuple` with 3 elements:

     [0]
         The data type as a :py:class:`numpy.dtype` object

     [1]
         The shape of the object, as a tuple of integers

     [2]
         The strides of the object, as a tuple of integers

   Returns ``0`` in case of failure, or a **new reference** to the tuple
   described above in case of success.


Bob File Support
----------------

.. cpp:type:: PyBobIoFileObject

   The pythonic object representation for a ``bob::io::File`` object.

   .. code-block:: cpp

      typedef struct {
        PyObject_HEAD
        boost::shared_ptr<bob::io::File> f;
      } PyBobIoFileObject;

   .. cpp:member:: boost::shared_ptr<bob::io::File> f

      A pointer to a file being read or written.

.. cpp:type:: PyBobIoFileIteratorObject

   The pythonic object representation for an iterator over a ``bob::io::File``
   object.

   .. code-block:: cpp

      typedef struct {
        PyObject_HEAD
        PyBobIoFileObject* pyfile;
        Py_ssize_t curpos;
      } PyBobIoFileIteratorObject;

   .. cpp:member:: PyBobIoFileObject* pyfile

      A pointer to the pythonic representation of a file.

   .. cpp:member:: Py_ssize_t curpos

      The current position at the file being pointed to.


Bob HDF5 Support
----------------

.. cpp:type:: PyBobIoHDF5FileObject

   The pythonic object representation for a ``bob::io::HDF5File`` object.

   .. code-block:: cpp

      typedef struct {
        PyObject_HEAD
        boost::shared_ptr<bob::io::HDF5File> f;
      } PyBobIoHDF5FileObject;

   .. cpp:member:: boost::shared_ptr<bob::io::HDF5File> f

      A pointer to a Bob object being used to read/write data into an HDF5
      file.


.. cpp:function:: int PyBobIoHDF5File_Check(PyObject* o)

   Checks if the input object ``o`` is a ``PyBobIoHDF5FileObject``. Returns
   ``1`` if it is, and ``0`` otherwise.


.. cpp:function:: int PyBobIoHDF5File_Converter(PyObject* o, PyBobIoHDF5FileObject** a)

   This function is meant to be used with :c:func:`PyArg_ParseTupleAndKeywords`
   family of functions in the Python C-API. It checks the input object to be of
   type ``PyBobIoHDF5FileObject`` and sets a **new reference** to it (in
   ``*a``) if it is the case. Returns ``0`` in case of failure, ``1`` in case
   of success.

Bob VideoReader Support
-----------------------

.. note::

   The video C-API (and Python) is only available if the package was compiled
   with FFMPEG or LibAV support.

.. cpp:type:: PyBobIoVideoReaderObject

   The pythonic object representation for a ``bob::io::VideoReader`` object.

   .. code-block:: cpp

      typedef struct {
        PyObject_HEAD
        boost::shared_ptr<bob::io::VideoReader> v;
      } PyBobIoVideoReaderObject;

   .. cpp:member:: boost::shared_ptr<bob::io::VideoReader> v

      A pointer to a Bob object being used to read the video contents

.. cpp:type:: PyBobIoVideoReaderIteratorObject

   The pythonic object representation for an iterator over a
   ``bob::io::VideoReader`` object.

   .. code-block:: cpp

      typedef struct {
        PyObject_HEAD
        PyBobIoVideoReaderObject* pyreader;
        boost::shared_ptr<bob::io::VideoReader::const_iterator> iter;
      } PyBobIoFileIteratorObject;

   .. cpp:member:: PyBobIoVideoReaderObject* pyreader

      A pointer to the pythonic representation of the video reader.

   .. cpp:member:: boost::shared_ptr<bob::io::VideoReader::const_iterator> iter

      The current position at the file being pointed to, represented by a
      formal iterator over the VideoReader.

.. cpp:type:: PyBobIoVideoReaderObject

   The pythonic object representation for a ``bob::io::VideoWriter`` object.

   .. code-block:: cpp

      typedef struct {
        PyObject_HEAD
        boost::shared_ptr<bob::io::VideoWriter> v;
      } PyBobIoVideoWriterObject;

   .. cpp:member:: boost::shared_ptr<bob::io::VideoWriter> v

      A pointer to a Bob object being used to write contents to the video.

.. include:: links.rst
