import math, os, pickle, re, string

class Bayes_Classifier:
    def __init__(self, trainDirectory = "reviews/"):
        '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text.'''

        self.positive = { "presence": 0, "frequency": {}, "bigrams": {} }
        self.negative = { "presence": 0, "frequency": {}, "bigrams": {} }

        if os.path.isfile("positive_best.pickle") and os.path.isfile("negative_best.pickle"):
            self.positive = self.load("positive_best.pickle")
            self.negative = self.load("negative_best.pickle")
        else:
            self.train(trainDirectory)

    def train(self, trainDirectory):
        '''Trains the Naive Bayes Sentiment Classifier.'''

        iFileList = []

        #accesses appropriate attribute depending on category
        table_wrapper = {
           '1': self.negative,
           '5': self.positive
        }

        for iFileObj in os.walk(trainDirectory):
            iFileList = iFileObj[2]
            break

        for file in iFileList:
            category = file.split('-')[1]

            #exclude files that are not positive or negative
            if category not in ["1", "5"]:
                continue

            contents = self.loadFile(trainDirectory + file)

            #strip punctuation and apply lowercase
            sanitized = ''.join(c for c in contents if c not in string.punctuation).lower()
            words = self.tokenize(sanitized)

            table = table_wrapper[category]
            table["presence"] += 1

            for i, word in enumerate(words):
                if word in table["frequency"]:
                    table["frequency"][word] += 1
                else:
                    table["frequency"][word] = 1

                #determine bigrams and store in seperate dict
                if i <= len(words) - 1:
                    bigram = ' '.join(words[i:i+2])
                    if bigram in table["bigrams"]:
                        table["bigrams"][bigram] += 1
                    else:
                        table["bigrams"][bigram] = 1

        self.save(self.positive, "positive_best.pickle")
        self.save(self.negative, "negative_best.pickle")

    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''

        table_wrapper = {
           "positive": self.positive,
           "negative": self.negative
        }

        total_documents = float(self.positive["presence"] + self.negative["presence"])
        len_vocab = len(self.positive["frequency"].keys()) + len(self.negative["frequency"].keys())

        negative_probability = 0
        positive_probability = 0

        #strip punctuation and lowercase everything
        sanitized = ''.join(c for c in sText if c not in string.punctuation).lower()
        words = self.tokenize(sanitized)

        for type in ["positive", "negative"]:

            dict = table_wrapper[type]
            sum_features = sum(dict["frequency"].values())
            probability = math.log(dict["presence"] / total_documents)
            test_freq = 1.0

            for i in range(1, len(words)):
                freq = 1.0

                bigram = ' '.join(words[i:i+2])

                if bigram in dict["bigrams"]:
                    freq += dict["bigrams"][bigram]

                #get count of first word in bigram in frequency table
                count = 0
                if words[i-1] in dict["frequency"]:
                    count += dict["frequency"][words[i-1]]

                probability += math.log(freq / (count + len_vocab))

            if type == "positive":
                positive_probability = probability
            else:
                negative_probability = probability


        if abs(positive_probability - negative_probability) <= 0.5:
            return "neutral"
        elif positive_probability > negative_probability:
            return "positive"
        else:
            return "negative"

    def loadFile(self, sFilename):
        '''Given a file name, return the contents of the file as a string.'''

        f = open(sFilename, "r")
        sTxt = f.read()
        f.close()
        return sTxt

    def save(self, dObj, sFilename):
        '''Given an object and a file name, write the object to the file using pickle.'''

        f = open(sFilename, "w")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()

    def load(self, sFilename):
        '''Given a file name, load and return the object stored in the file.'''

        f = open(sFilename, "r")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        '''Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order).'''

        lTokens = []
        sToken = ""
        for c in sText:
           if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
              sToken += c
           else:
              if sToken != "":
                 lTokens.append(sToken)
                 sToken = ""
              if c.strip() != "":
                 lTokens.append(str(c.strip()))

        if sToken != "":
           lTokens.append(sToken)

        return lTokens
