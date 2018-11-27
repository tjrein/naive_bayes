==== CS510 HW2 ====
    Author: Tom Rein
    Email: tr557@drexel.edu

==== File Contents ====
    * bayes.py
        - Original unigram Bayes classifier

    * bayesbest.py
        - Improved Bayes classifier.
        - Classifies using bigrams.
        - Normalizes data striping punctuation and applying lowercase
        - imports the native string module as an added dependency

    * evaluate_document.pdf
        - Contains answers to reflection questions
        - Outlines steps taken to improve classifier
        - Compares both classifiers using precision, recall, and f- measure
        - Offers several future ways to potentially extend performance

    * evaluate.py
        - executable file used to train and test classifiers
        - prints categorization results of running classifier on test data folder
        - I was not if evaluate.py was intended for internal testing purposes, so I included it just in case.

==== Setup and Assumptions ===

    * Pickle files

        When initialized, the bayes.py classifier will attempt to load pickle files named "positive.pickle" and "negative.pickle".
        Similarly, the bayesbest.py classifier will attempt to load pickle filed named "positive_best.pickle" and "negative_best.pickle"

        IF the associative pickle files are not found, both classifiers will create them from the training data.
        Pickle files are expected to be housed in the same directory as the classifiers.

    * Training data

        Training data can be sent to a classifier on init by passing in a folder as an argument.
            ex. b = bayes.Bayes_Classifier("training/")

        This classifier will walk through the files contained in "training/", train the system, and pickle the resulting dicts.
        If no folder is passed as an argument, a default value of "reviews/" will be assigned.
        Note that a trailing slash is expected.

        With respect to evaluate.py, this can be further configured by modifying the value of the variable TrainDir.
        evaluate.py will initialize the classifier with whatever value is stored in TrainDir

    * Changing classifiers in evaluate.py
        Determining which classifier to initialize will depend on the value of the variable testFile.
        This value will need to be updated manually to either be "bayes.py" or "bayesbest.py" to test the respective classifier.
        The initial value is set to "bayes.py"

    * Test data
        evaluate.py has been modified to accept an optional command line argument.
        This argument is assumed to be the name of a folder, and will replace the "testDir" variable with the supplied value.

==== Usage Instructions ====

    * Commands To train and test classifiers using evaluate.py:
       - "python evaluate.py"
       - "python evaluate.py [nameOfTestFolder]"

       The first command will use the value of "trainDir" as training data
       The second command will replace the value of "trainDir" with [nameOfTestFolder] and use the new folder to train

    * Usage in a python shell:
       >>> import bayesbest.py
       >>> b = bayesbest.Bayes_Classifier("myTrainingFiles/")
       >>> result = b.classify("I love my AI class")
       >>> print result
