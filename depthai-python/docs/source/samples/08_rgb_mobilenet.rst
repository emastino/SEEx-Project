08 - RGB & MobilenetSSD
=======================

This example shows how to run MobileNetv2SSD on the RGB input frame, and how to display both the RGB
preview and the metadata results from the MobileNetv2SSD on the preview.

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/68VkKGO1WX4" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

This example also requires MobilenetSDD blob (:code:`mobilenet.blob` file) to work - you can download it from
`here <https://artifacts.luxonis.com/artifactory/luxonis-depthai-data-local/network/mobilenet-ssd_openvino_2021.2_6shave.blob>`__

Source code
###########

Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/08_rgb_mobilenet.py>`__

.. literalinclude:: ../../../examples/08_rgb_mobilenet.py
   :language: python
   :linenos:

.. include::  /includes/footer-short.rst
