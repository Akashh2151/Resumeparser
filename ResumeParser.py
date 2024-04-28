import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

# print(spacy.__version__)

import json

with open(r'C:\Users\EMC\Desktop\resume\dataset\dataset.json', 'r') as file:
    cv_data = json.load(file)
# print(len(cv_data))
# print(cv_data[0])


def get_spacy_doc(file,data):
    nlp=spacy.blank("en")
    db=DocBin()
    
    for text,annot in tqdm(data):
        doc=nlp.make_doc(text)
        annot=annot['entities']
        
        ents=[]
        entity_indices=[]
        
        for start,end,label in annot:
            skip_entity=False
            for idx in range(start,end):
                if idx in entity_indices:
                    skip_entity=True
                    break
                if skip_entity == True:
                    continue
                
                entity_indices= entity_indices+list(range(start,end))
                
                try:
                    span=doc.char_span(start,end,label=label,alignment_mode='strict')
                except:
                    continue
                
                if span is None:
                    err_data=str([start,end])+"    " + str(text) + "\n" 
                    file.write(err_data)
                    
                else:
                    ents.append(span)
            try:
                doc.ents=ents
                db.add(doc)
            except:
                pass
                           
        return db
    
train,test=train_test_split(cv_data,test_size=0.3)
# print(len(train),len(test))


file = open(r'C:\Users\EMC\Desktop\resume\model\train_file.txt', 'w')


db=get_spacy_doc(file,train)
db.to_disk(r'C:\Users\EMC\Desktop\resume\model\train_data.spacy')


db = get_spacy_doc(file,test)
db.to_disk(r'C:\Users\EMC\Desktop\resume\model\test_data.spacy')


file.close()

#  C:\Users\EMC\Desktop\resume> python -m spacy train C:\Users\EMC\Desktop\resume\config\config.cfg --output C:\Users\EMC\Desktop\resume\model\output --paths.train C:\Users\EMC\Desktop\resume\model\train_data.spacy --paths.dev C:\Users\EMC\Desktop\resume\model\test_data.spacy

