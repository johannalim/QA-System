import openai
import numpy as np
import pandas as pd
import sys, getopt
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
import re
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

# insert ChatGPT OpenAI key here
# OPENAI_API_KEY = ""

file = sys.argv[1]

reader = PdfReader(file)

# extracting text from page
text = [reader.pages[i].extract_text() for i in range(len(reader.pages))]

holdtext = " ".join(text)
textandref = holdtext.split("References")
full_text, references = textandref[0], textandref[1]


paragraphs = full_text.split(".\n")
temp = [section.split("-\n") for section in paragraphs]
para_lines = []


for para in temp:
    joinline = "".join(para)
    splitjoin = joinline.split("\n")
    join2 = " ".join(splitjoin)
    hold = join2.split(". ")
    for s in hold:
        sentences = s.split(". ")
        para_lines.append([sent for sent in sentences if sent != '' and not sent.isnumeric()])


reftemp = references.split(".\n")
ref_lines = []

for s in reftemp:
    l = s.split("\n")
    sent = "".join(l)
    if sent != '' and not sent.isnumeric(): ref_lines.append(sent)

ref_tokenized = [word_tokenize(lines) for lines in ref_lines]

sw_list = list(stopwords.words("english"))
sw_list.append(",")
sw_list.append(".")
sw_list.append("[")
sw_list.append("]")
ref_tokens_swr = list()

for p in ref_tokenized:
    line = [token for token in p if token.lower() not in sw_list]
    ref_tokens_swr.append(line)


tokenized = [[word_tokenize(lines) for lines in para] for para in para_lines]
word_count = {}
word_set = []
index_dict = {}
i = 0
 
for p in tokenized:
    for l in p:
        for w in l:
            word = w.lower()
            if word not in word_set:
                word_set.append(word)
                word_count[word] = 1
                index_dict[word] = i
                i += 1
            else:
                word_count[word] += 1

embeddings_dict = {}
with open("glove.840B.300d.txt", 'r', encoding="utf-8") as f:
    print("Loading pre-trained GloVe embeddings . . . ")
    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        vector = np.asarray(values[-300:], "float32")
        embeddings_dict[word] = vector
    print("Load complete!")

# sentence embedding
def DWfreqEmbed(sentence):
    x = [word.lower() for word in word_tokenize(sentence) if word.isalpha()]
    dwEmbed = []
    for word in x:
        if word in word_set:
            freq = word_count[word]
            dw = 1/freq
            if word in embeddings_dict: 
                embed = embeddings_dict[word]
                dwE = [dw*e for e in embed]
                dwEmbed.append(dwE)
        elif word not in sw_list:
            if word in embeddings_dict: 
                embed = embeddings_dict[word]
                dwEmbed.append(embed)
        if word in ref_tokens_swr:
            embed = embeddings_dict["reference"]
            refw = word_count[word]/10
            dwE = [refw*e for e in embed]
            dwEmbed.append(dwE)
    
    if len(dwEmbed) > 0: finalEmbed = np.array(dwEmbed).mean(axis=0)
    else: finalEmbed = np.zeros(embeddings_dict['0'].shape)

    return finalEmbed

def dataDWEmbed(para_lines):
    dataembeds = {}
    for para in para_lines:
        paragraph = " ".join(para)
        paraEmbed = {}
        for lines in para:
            embed = DWfreqEmbed(lines)
            paraEmbed[lines] = embed
        dataembeds[paragraph] = paraEmbed
    return dataembeds

pdfEmbed = dataDWEmbed(para_lines)

print("\nEnter your query: ")
query = input()


def cosSim(query,pdfEmbed):
    sentEmbed = DWfreqEmbed(query)
    sentEmbed = sentEmbed.reshape(1,-1)

    diff = {}
    for p in pdfEmbed.keys():
        k = pdfEmbed[p].keys()
        for i in k:
            emb = pdfEmbed[p][i]
            emb = emb.reshape(1,-1)
            sim = cosine_similarity(emb,sentEmbed)
            add = np.nansum(abs(sim))
            if add != 0: diff[i] = add

    closest = sorted(diff.items(), key=lambda x: x[1])
    closest = dict(closest[-30:])
    ans = list(closest.keys())
    return ans


closest_vectors = cosSim(query,pdfEmbed)
closest_text = " ".join(closest_vectors)

x = [word for word in word_tokenize(closest_text) if (word.isalnum() or re.search('-|,|\+|:', word) != None) and word.lower() not in sw_list]
closest_text = " ".join(x)

encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_tokens = len(encoding.encode(closest_text))
while num_tokens > 766: 
    diff = num_tokens - 766
    if diff == 0: diff = 2
    x = [word for word in word_tokenize(closest_text)]
    closest_text = " ".join(x[diff:])
    num_tokens = len(encoding.encode(closest_text))

openai.api_key = OPENAI_API_KEY
response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    temperature = 0.5,
    messages = [
        {"role": "system", "content": "Answer questions using assistant content extracted from the paper"},
        {"role": "assistant", "content": closest_text},
        {"role": "user", "content": query}
    ]
)
print(response.choices[0].message.content)

print("Tokens in input content:", num_tokens)
print("Tokens taken by ChatGPT API:", response.usage.prompt_tokens)