from openai import OpenAI
client = OpenAI()



InsertcontextualWordEmbeddings = open('/Users/biswas/Hredoy/Research/Data Augmentation/GPT_DataSet/GPT New Prompt Data/Without Label/TrainingDataSet_97_Augmented_Generated_Keyword.txt', 'w+')

with open('/Users/biswas/Hredoy/Research/Data Augmentation/GPT_DataSet/GPT New Prompt Data/Without Label/Line_Original_Negative_Only_TrainingDataSet_97.txt', 'r') as f:
     Lines = f.readlines()
     counter = 0
     for eline in Lines:
#           line = eline.split("\t")
#           tweet = line[0]
          completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content":eline},
    {"role": "user", "content": "Can you generate five different paraphrases of this original tweet? Paraphrases cannot use the words from the original tweet. But you need to preserve the meaning and the writing style of the original tweets. Paraphrases also need to be different from each other. "}
  ]
)


         # print(type(completion.choices[0].message))
          InsertcontextualWordEmbeddings.write(str(completion.choices[0].message.content))
#           print(str(completion.choices[0].message.content))
#           print(str(completion.choices[1].message.content))
#           InsertcontextualWordEmbeddings.write("\t" + line[1])
          InsertcontextualWordEmbeddings.write("\n")
          #counter = counter + 1
          #if (counter == 3):
           #   break

InsertcontextualWordEmbeddings.close()
