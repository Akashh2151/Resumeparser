import streamlit as st
import spacy
import fitz  # PyMuPDF
import re  # Regular expression library

# Load the pre-trained Spacy model
nlp = spacy.load(r'C:\Akash\ResumeParser\model\output\model-best')

def extract_text_from_pdf(file):
    file_stream = file.read()
    doc = fitz.open(stream=file_stream, filetype="pdf")
    text = " "
    for page in doc:
        text += page.get_text()
    return text

def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = ent.text
        else:
            if ent.text not in entities[ent.label_]:
                entities[ent.label_] += "; " + ent.text

    # Refine checks for email and phone by combining similar entity types
    email_key = 'EMAIL' if 'EMAIL' in entities else 'EMAIL ADDRESS' if 'EMAIL ADDRESS' in entities else None
    if not email_key:
        # Extract email using regex if not found by the model
        emails = extract_emails(text)
        if emails:
            entities['EMAIL'] = emails

    phone_key = 'PHONE' if 'PHONE' in entities else 'PHONE NUMBER' if 'PHONE NUMBER' in entities else None
    if not phone_key:
        # Extract phone numbers using regex if not found by the model
        phones = extract_phones(text)
        if phones:
            entities['PHONE'] = phones

    print(entities)
    return entities

def extract_emails(text):
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    emails = '; '.join(set(re.findall(email_regex, text)))
    return emails

def extract_phones(text):
    phone_regex = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    phones = '; '.join(set(re.findall(phone_regex, text)))
    return phones

def main():
    st.title("Resume Information Extractor")
    uploaded_file = st.sidebar.file_uploader("Upload your resume in PDF format", type=["pdf", "docx"])
    
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        entities = extract_entities(text)

        if entities:
            for label, value in entities.items():
                st.subheader(f"{label.replace('_', ' ').title()}:")
                st.write(value)

if __name__ == "__main__":
    main()
