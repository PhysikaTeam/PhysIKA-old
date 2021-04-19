Object Categorization
=====================

.. highlight:: cpp

This section describes approaches based on local 2D features and used to categorize objects.

.. note::

   * A complete Bag-Of-Words sample can be found at opencv_source_code/samples/cpp/bagofwords_classification.cpp

   * (Python) An example using the features2D framework to perform object categorization can be found at opencv_source_code/samples/python2/find_obj.py

BOWTrainer
----------
.. ocv:class:: BOWTrainer

Abstract base class for training the *bag of visual words* vocabulary from a set of descriptors.
For details, see, for example, *Visual Categorization with Bags of Keypoints* by Gabriella Csurka, Christopher R. Dance,
Lixin Fan, Jutta Willamowski, Cedric Bray, 2004. ::

    class BOWTrainer
    {
    public:
        BOWTrainer(){}
        virtual ~BOWTrainer(){}

        void add( const Mat& descriptors );
        const vector<Mat>& getDescriptors() const;
        int descripotorsCount() const;

        virtual void clear();

        virtual Mat cluster() const = 0;
        virtual Mat cluster( const Mat& descriptors ) const = 0;

    protected:
        ...
    };

BOWTrainer::add
-------------------
Adds descriptors to a training set.

.. ocv:function:: void BOWTrainer::add( const Mat& descriptors )

    :param descriptors: Descriptors to add to a training set. Each row of  the ``descriptors``  matrix is a descriptor.

The training set is clustered using ``clustermethod`` to construct the vocabulary.

BOWTrainer::getDescriptors
------------------------------
Returns a training set of descriptors.

.. ocv:function:: const vector<Mat>& BOWTrainer::getDescriptors() const



BOWTrainer::descripotorsCount
---------------------------------
Returns the count of all descriptors stored in the training set.

.. ocv:function:: int BOWTrainer::descripotorsCount() const



BOWTrainer::cluster
-----------------------
Clusters train descriptors.

.. ocv:function:: Mat BOWTrainer::cluster() const

.. ocv:function:: Mat BOWTrainer::cluster( const Mat& descriptors ) const

    :param descriptors: Descriptors to cluster. Each row of  the ``descriptors``  matrix is a descriptor. Descriptors are not added to the inner train descriptor set.

The vocabulary consists of cluster centers. So, this method returns the vocabulary. In the first variant of the method, train descriptors stored in the object are clustered. In the second variant, input descriptors are clustered.

BOWKMeansTrainer
----------------
.. ocv:class:: BOWKMeansTrainer : public BOWTrainer

:ocv:func:`kmeans` -based class to train visual vocabulary using the *bag of visual words* approach.
::

    class BOWKMeansTrainer : public BOWTrainer
    {
    public:
        BOWKMeansTrainer( int clusterCount, const TermCriteria& termcrit=TermCriteria(),
                          int attempts=3, int flags=KMEANS_PP_CENTERS );
        virtual ~BOWKMeansTrainer(){}

        // Returns trained vocabulary (i.e. cluster centers).
        virtual Mat cluster() const;
        virtual Mat cluster( const Mat& descriptors ) const;

    protected:
        ...
    };

BOWKMeansTrainer::BOWKMeansTrainer
----------------------------------

The constructor.

.. ocv:function:: BOWKMeansTrainer::BOWKMeansTrainer( int clusterCount, const TermCriteria& termcrit=TermCriteria(), int attempts=3, int flags=KMEANS_PP_CENTERS )

    See :ocv:func:`kmeans` function parameters.

BOWImgDescriptorExtractor
-------------------------
.. ocv:class:: BOWImgDescriptorExtractor

Class to compute an image descriptor using the *bag of visual words*. Such a computation consists of the following steps:

    #. Compute descriptors for a given image and its keypoints set.
    #. Find the nearest visual words from the vocabulary for each keypoint descriptor.
    #. Compute the bag-of-words image descriptor as is a normalized histogram of vocabulary words encountered in the image. The ``i``-th bin of the histogram is a frequency of ``i``-th word of the vocabulary in the given image.

The class declaration is the following: ::

        class BOWImgDescriptorExtractor
        {
        public:
            BOWImgDescriptorExtractor( const Ptr<DescriptorExtractor>& dextractor,
                                       const Ptr<DescriptorMatcher>& dmatcher );
            virtual ~BOWImgDescriptorExtractor(){}

            void setVocabulary( const Mat& vocabulary );
            const Mat& getVocabulary() const;
            void compute( const Mat& image, vector<KeyPoint>& keypoints,
                          Mat& imgDescriptor,
                          vector<vector<int> >* pointIdxsOfClusters=0,
                          Mat* descriptors=0 );
            int descriptorSize() const;
            int descriptorType() const;

        protected:
            ...
        };




BOWImgDescriptorExtractor::BOWImgDescriptorExtractor
--------------------------------------------------------
The constructor.

.. ocv:function:: BOWImgDescriptorExtractor::BOWImgDescriptorExtractor(           const Ptr<DescriptorExtractor>& dextractor,          const Ptr<DescriptorMatcher>& dmatcher )

    :param dextractor: Descriptor extractor that is used to compute descriptors for an input image and its keypoints.

    :param dmatcher: Descriptor matcher that is used to find the nearest word of the trained vocabulary for each keypoint descriptor of the image.



BOWImgDescriptorExtractor::setVocabulary
--------------------------------------------
Sets a visual vocabulary.

.. ocv:function:: void BOWImgDescriptorExtractor::setVocabulary( const Mat& vocabulary )

    :param vocabulary: Vocabulary (can be trained using the inheritor of  :ocv:class:`BOWTrainer` ). Each row of the vocabulary is a visual word (cluster center).



BOWImgDescriptorExtractor::getVocabulary
--------------------------------------------
Returns the set vocabulary.

.. ocv:function:: const Mat& BOWImgDescriptorExtractor::getVocabulary() const



BOWImgDescriptorExtractor::compute
--------------------------------------
Computes an image descriptor using the set visual vocabulary.

.. ocv:function:: void BOWImgDescriptorExtractor::compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& imgDescriptor, vector<vector<int> >* pointIdxsOfClusters=0, Mat* descriptors=0 )

    :param image: Image, for which the descriptor is computed.

    :param keypoints: Keypoints detected in the input image.

    :param imgDescriptor: Computed output image descriptor.

    :param pointIdxsOfClusters: Indices of keypoints that belong to the cluster. This means that ``pointIdxsOfClusters[i]``  are keypoint indices that belong to the  ``i`` -th cluster (word of vocabulary) returned if it is non-zero.

    :param descriptors: Descriptors of the image keypoints  that are returned if they are non-zero.



BOWImgDescriptorExtractor::descriptorSize
---------------------------------------------
Returns an image descriptor size if the vocabulary is set. Otherwise, it returns 0.

.. ocv:function:: int BOWImgDescriptorExtractor::descriptorSize() const



BOWImgDescriptorExtractor::descriptorType
---------------------------------------------

Returns an image descriptor type.

.. ocv:function:: int BOWImgDescriptorExtractor::descriptorType() const
