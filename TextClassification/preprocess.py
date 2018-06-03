import codecs

with codecs.open('15000tweets.csv', 'w', 'utf-8') as outfile:
	with codecs.open('Sentiment Analysis Dataset.csv', 'r', 'utf-8') as infile:
		for i, line in enumerate(infile):
			outfile.write(line)
			if i == 15000:
				break
