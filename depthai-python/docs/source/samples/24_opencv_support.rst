24 - OpenCV support
=========================

This example shows API which exposes both numpy and OpenCV compatible image types for eaiser usage.
It uses ColorCamera node to retrieve both BGR interleaved 'preview' and NV12 encoded 'video' frames.
Both are displayed using functions `getFrame` and `getCvFrame`.

Setup
#####

.. include::  /includes/install_from_pypi.rst

This example also requires MobilenetSDD blob (:code:`mobilenet.blob` file) to work - you can download it from
`here <https://artifacts.luxonis.com/artifactory/luxonis-depthai-data-local/network/mobilenet.blob>`__

Source code
###########

Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/24_opencv_support.py>`__

.. literalinclude:: ../../../examples/24_opencv_support.py
   :language: python
   :linenos:

.. include::  /includes/footer-short.rst
