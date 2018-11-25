import math, os, pickle, re

class Bayes_Classifier:
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

      path = "movie_and_product_reviews/db_txt_files"
      #path = "movies_reviews/movies_reviews"

      table_wrapper = {
         '1': self.negative,
         '5': self.positive
      }

      for iFileObj in os.walk(path):
          iFileList = iFileObj[2]
          break

      for file in iFileList:
          category = file.split('-')[1]
          if category not in ["1", "5"]:
              continue
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

      table_wrapper = {
         "positive": self.positive,
         "negative": self.negative
      }

      total_documents = float(self.positive["presence"] + self.negative["presence"])

      negative_probability = 0
      positive_probability = 0

      words = self.tokenize(sText)

      for type in ["positive", "negative"]:

          dict = table_wrapper[type]

          prior_probability = math.log(dict["presence"] / total_documents)
          sum_features = float(sum(dict["frequency"].values()))
          probability = 0

          for word in words:
              freq = 1

              if word in dict["frequency"]:
                  freq += dict["frequency"][word]

              probability += math.log(freq / sum_features)

              if type == "positive":
                  positive_probability = probability
              else:
                  negative_probability = probability

      print "test_positive", positive_probability
      print "test_negative", negative_probability

      if positive_probability > negative_probability:
         return "positive"
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
