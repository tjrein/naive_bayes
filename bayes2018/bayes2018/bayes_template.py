import math, os, pickle, re

class Bayes_Classifier:
   #TEST SYMLINK
   def __init__(self, trainDirectory = "movie_reviews/"):
      '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
      cache of a trained classifier has been stored, it loads this cache.  Otherwise,
      the system will proceed through training.  After running this method, the classifier
      is ready to classify input text.'''

      self.positive = { "presence": 0, "frequency": {} }
      self.negative = { "presence": 0, "frequency": {} }

      if os.path.isfile("positive.pickle") and os.path.isfile("negative.pickle"):
          self.positive = self.load("positive.pickle")
          self.negative = self.load("negative.pickle")
      else:
          self.train()

   def train(self):
      '''Trains the Naive Bayes Sentiment Classifier.'''

      iFileList = []
      path = "movies_reviews/movies_reviews"

      table_wrapper = {
         '1': self.negative,
         '5': self.positive
      }

      for iFileObj in os.walk(path):
          iFileList = iFileObj[2]
          break

      for file in iFileList:
          category = file.split('-')[1]
          contents = self.loadFile(path + '/' + file)
          words = self.tokenize(contents)
          table = table_wrapper[category]

          table["presence"] += 1

          for word in words:
              if word in table["frequency"]:
                  table["frequency"][word] += 1
              else:
                  table["frequency"][word] = 1

      self.save(self.positive, "positive.pickle")
      self.save(self.negative, "negative.pickle")

   def classify(self, sText):
      '''Given a target string sText, this function returns the most likely document
      class to which the target string belongs. This function should return one of three
      strings: "positive", "negative" or "neutral".
      '''

      total_documents = float(self.positive["presence"] + self.negative["presence"])
      prior_probability = self.positive["presence"] / total_documents

      sum_positive_features = float(sum(self.positive["frequency"].values()))
      sum_negative_features = float(sum(self.negative["frequency"].values()))
      words = self.tokenize(sText)


      print self.positive["frequency"].values()

      print sum_negative_features
      print sum_positive_features


      return

      positive_probability = 0
      for word in words:
          freq = 1

          if word in self.positive["frequency"]:
              freq += self.positive["frequency"][word]

          positive_probability += math.log(freq / sum_positive_features)

      negative_probability = 0
      for word in words:
          freq = 1

          if word in self.positive["frequency"]:
              freq += self.positive["frequency"][word]

          negative_probability += math.log(freq / sum_negative_features)




      print "positive", positive_probability
      print "negative", negative_probability


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
