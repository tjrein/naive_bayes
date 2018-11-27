import sys

testFile = "bayesbest.py"
trainDir = "training/"
testDir = "testing/"

#check to see if filename is based via command line
if len(sys.argv) == 2:
	testDir = sys.argv[1]

    #append a trailing slash if it not the last character
	if testDir[-1] != '/':
		testDir += '/'

execfile(testFile)
bc = Bayes_Classifier(trainDir)

iFileList = []

for fFileObj in os.walk(testDir):
	iFileList = fFileObj[2]
	break
print '%d test reviews.' % len(iFileList)

results = {"negative":0, "neutral":0, "positive":0}

print "\nFile Classifications:"
for filename in iFileList:
	fileText = bc.loadFile(testDir + filename)
	result = bc.classify(fileText)
	print "%s: %s" % (filename, result)
	results[result] += 1

print "\nResults Summary:"
for r in results:
	print "%s: %d" % (r, results[r])
