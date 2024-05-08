import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
import sys,fitz

print(spacy.__version__)

# import json

with open(r'C:\Akash\ResumeParser\dataset\annotations.json', 'r') as file:
    cv_data = json.load(file)
# print(len(cv_data))
# print(cv_data[0])


def get_spacy_doc(file, data):
    nlp = spacy.blank("en")
    db = DocBin()

    for text, annot in tqdm(data):
        doc = nlp.make_doc(text)
        annot = annot['entities']

        ents = []
        entity_indices = set()  # Using a set to track indices for better performance

        for start, end, label in annot:
            if not any(idx in entity_indices for idx in range(start, end)):  # Check for overlap
                entity_indices.update(range(start, end))
                span = doc.char_span(start, end, label=label, alignment_mode='strict')
                if span is None:
                    file.write(f"Error: Unable to form span for {start}-{end} in text: {text[:50]}...\n")
                else:
                    ents.append(span)

        doc.ents = ents  # Set the document entities
        db.add(doc)  # Add the processed document to DocBin

    return db
    
train,test=train_test_split(cv_data,test_size=0.3)
# print(len(train),len(test))


file = open(r'C:\Akash\ResumeParser\model\train_file.txt', 'w')


db=get_spacy_doc(file,train)
db.to_disk(r'C:\Akash\ResumeParser\model\train_data.spacy')


db = get_spacy_doc(file,test)
db.to_disk(r'C:\Akash\ResumeParser\model\test_data.spacy')


file.close()


# python -m spacy init fill-config C:\Akash\ResumeParser\config\base_config.cfg C:\Akash\ResumeParser\config\config.cfg
     
# python -m spacy train C:\Akash\ResumeParser\config\config.cfg --output C:\Akash\ResumeParser\model\output --paths.train C:\Akash\ResumeParser\model\train_data.spacy --paths.dev C:\Akash\ResumeParser\model\test_data.spacy --gpu-id 0


# nlp=spacy.load(r'C:\Users\EMC\Desktop\resume\model\output\model-best')
# fname=r'C:\Users\EMC\Desktop\resume\test\AkashDesai.pdf'
# doc=fitz.open(fname)

# text= "  "
# for page in doc:
#     text= text + str(page.get_text())

# # print(test)

# doc = nlp(text)
# for ent in doc.ents:
#     print(ent,"->>>>>>>>>>",ent.label_)
    


