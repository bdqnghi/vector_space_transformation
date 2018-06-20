import os

projects = ["antlr","cordova","datastax","factual","fpml","log4j","spring","lucene","uap","zeromq","itext","jgit","poi","jts","db4o"]
CURRENT_DIR = os.getcwd()

cs_sentences = list()
java_sentences = list()
for project in projects:
	for r,ds,files in os.walk(os.path.join(CURRENT_DIR,"DATA",project)):
		for file in files:

			file_path = os.path.join(r,file)
			splits = file_path.split("/")
			with open(file_path,"r") as f:
				sentence = f.read()

			if sentence:
				if splits[7] == "cs":
					# cs_sentences.append(sentence)
					with open("cs_sentences.txt","a") as f:
						f.write(sentence)
						f.write("\n")
				if splits[7] == "java":
					# java_sentences.append(sentence)
					with open("java_sentences.txt","a") as f:
						f.write(sentence)
						f.write("\n")
