.. _InteroperabilityWithOpenCV1:

Interoperability with OpenCV 1
******************************

Goal
====

For the OpenCV developer team it's important to constantly improve the library. We are constantly thinking about methods that will ease your work process, while still maintain the libraries flexibility. The new C++ interface is a development of us that serves this goal. Nevertheless, backward compatibility remains important. We do not want to break your code written for earlier version of the OpenCV library. Therefore, we made sure that we add some functions that deal with this. In the following you'll learn:

.. container:: enumeratevisibleitemswithsquare

   + What changed with the version 2 of OpenCV in the way you use the library compared to its first version
   + How to add some Gaussian noise to an image
   + What are lookup tables and why use them?

General
=======

When making the switch you first need to learn some about the new data structure for images: :ref:`matTheBasicImageContainer`, this replaces the old *CvMat* and *IplImage* ones. Switching to the new functions is easier. You just need to remember a couple of new things.

OpenCV 2 received reorganization. No longer are all the functions crammed into a single library. We have many modules, each of them containing data structures and functions relevant to certain tasks. This way you do not need to ship a large library if you use just a subset of OpenCV. This means that you should also include only those headers you will use. For example:

.. code-block:: cpp

   #include <opencv2/core/core.hpp>
   #include <opencv2/imgproc/imgproc.hpp>
   #include <opencv2/highgui/highgui.hpp>


All the OpenCV related stuff is put into the *cv* namespace to avoid name conflicts with other libraries data structures and functions. Therefore, either you need to prepend the *cv::* keyword before everything that comes from OpenCV or after the includes, you just add a directive to use this:

.. code-block:: cpp

   using namespace cv;  // The new C++ interface API is inside this namespace. Import it.

Because the functions are already in a namespace there is no need for them to contain the *cv* prefix in their name. As such all the new C++ compatible functions don't have this and they follow the camel case naming rule. This means the first letter is small (unless it's a name, like Canny) and the subsequent words start with a capital letter (like *copyMakeBorder*).

Now, remember that you need to link to your application all the modules you use, and in case you are on Windows using the *DLL* system you will need to add, again, to the path all the binaries. For more in-depth information if you're on Windows read :ref:`Windows_Visual_Studio_How_To` and for Linux an example usage is explained in :ref:`Linux_Eclipse_Usage`.

Now for converting the *Mat* object you can use either the *IplImage* or the *CvMat* operators. While in the C interface you used to work with pointers here it's no longer the case. In the C++ interface we have mostly *Mat* objects. These objects may be freely converted to both *IplImage* and *CvMat* with simple assignment. For example:

.. code-block:: cpp

   Mat I;
   IplImage pI = I;
   CvMat    mI = I;

Now if you want pointers the conversion gets just a little more complicated. The compilers can no longer automatically determinate what you want and as you need to explicitly specify your goal. This is to call the *IplImage* and *CvMat* operators and then get their pointers. For getting the pointer we use the & sign:

.. code-block:: cpp

   Mat I;
   IplImage* pI     = &I.operator IplImage();
   CvMat* mI        =  &I.operator CvMat();

One of the biggest complaints of the C interface is that it leaves all the memory management to you. You need to figure out when it is safe to release your unused objects and make sure you do so before the program finishes or you could have troublesome memory leeks. To work around this issue in OpenCV there is introduced a sort of smart pointer. This will automatically release the object when it's no longer in use. To use this declare the pointers as a specialization of the *Ptr* :

.. code-block:: cpp

   Ptr<IplImage> piI = &I.operator IplImage();

Converting from the C data structures to the *Mat* is done by passing these inside its constructor. For example:

.. code-block:: cpp

   Mat K(piL), L;
   L = Mat(pI);

A case study
============

Now that you have the basics done :download:`here's <../../../../samples/cpp/tutorial_code/core/interoperability_with_OpenCV_1/interoperability_with_OpenCV_1.cpp>` an example that mixes the usage of the C interface with the C++ one. You will also find it in the sample directory of the OpenCV source code library at the :file:`samples/cpp/tutorial_code/core/interoperability_with_OpenCV_1/interoperability_with_OpenCV_1.cpp` . To further help on seeing the difference the programs supports two modes: one mixed C and C++ and one pure C++. If you define the *DEMO_MIXED_API_USE* you'll end up using the first. The program separates the color planes, does some modifications on them and in the end merge them back together.


.. literalinclude:: ../../../../samples/cpp/tutorial_code/core/interoperability_with_OpenCV_1/interoperability_with_OpenCV_1.cpp
   :language: cpp
   :linenos:
   :tab-width: 4
   :lines: 1-9, 22-25, 27-44

Here you can observe that with the new structure we have no pointer problems, although it is possible to use the old functions and in the end just transform the result to a *Mat* object.

.. literalinclude:: ../../../../samples/cpp/tutorial_code/core/interoperability_with_OpenCV_1/interoperability_with_OpenCV_1.cpp
   :language: cpp
   :linenos:
   :tab-width: 4
   :lines: 46-51

Because, we want to mess around with the images luma component we first convert from the default BGR to the YUV color space and then split the result up into separate planes. Here the program splits: in the first example it processes each plane using one of the three major image scanning algorithms in OpenCV (C [] operator, iterator, individual element access). In a second variant we add to the image some Gaussian noise and then mix together the channels according to some formula.

The scanning version looks like:

.. literalinclude:: ../../../../samples/cpp/tutorial_code/core/interoperability_with_OpenCV_1/interoperability_with_OpenCV_1.cpp
   :language: cpp
   :linenos:
   :tab-width: 4
   :lines: 55-75

Here you can observe that we may go through all the pixels of an image in three fashions: an iterator, a C pointer and an individual element access style. You can read a more in-depth description of these in the :ref:`howToScanImagesOpenCV` tutorial. Converting from the old function names is easy. Just remove the cv prefix and use the new *Mat* data structure. Here's an example of this by using the weighted addition function:

.. literalinclude:: ../../../../samples/cpp/tutorial_code/core/interoperability_with_OpenCV_1/interoperability_with_OpenCV_1.cpp
   :language: cpp
   :linenos:
   :tab-width: 4
   :lines: 79-112

As you may observe the *planes* variable is of type *Mat*. However, converting from *Mat* to *IplImage* is easy and made automatically with a simple assignment operator.

.. literalinclude:: ../../../../samples/cpp/tutorial_code/core/interoperability_with_OpenCV_1/interoperability_with_OpenCV_1.cpp
   :language: cpp
   :linenos:
   :tab-width: 4
   :lines: 115-127

The new *imshow* highgui function accepts both the *Mat* and *IplImage* data structures. Compile and run the program and if the first image below is your input you may get either the first or second as output:

.. image:: images/outputInteropOpenCV1.jpg
   :alt: The output of the sample
   :align: center


You may observe a runtime instance of this on the `YouTube here <https://www.youtube.com/watch?v=qckm-zvo31w>`_ and you can :download:`download the source code from here <../../../../samples/cpp/tutorial_code/core/interoperability_with_OpenCV_1/interoperability_with_OpenCV_1.cpp>` or find it in the :file:`samples/cpp/tutorial_code/core/interoperability_with_OpenCV_1/interoperability_with_OpenCV_1.cpp` of the OpenCV source code library.

.. raw:: html

  <div align="center">
  <iframe title="Interoperability with OpenCV 1" width="560" height="349" src="http://www.youtube.com/embed/qckm-zvo31w?rel=0&loop=1" frameborder="0" allowfullscreen align="middle"></iframe>
  </div>
