# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:59:28 2020

@author: DELL
"""

# import fasttext
from transformers import MarianTokenizer, MarianMTModel
import streamlit as st

st.title('Danish <--> English Translator')
text = st.text_area("Enter Text:", value='', height=None, max_chars=None, key=None)

#text = "Der er i weekenden gennemført ændring på nedenstående fællesdrev (file shares). Ændringen er ved en fejl ikke blevet kommunikeret ordentligt."
# Answer: There has been a change on the common drive (file shares) below this weekend. The change has not been communicated properly by mistake.
#text = "Der er ændring på hvilken sti (path), som skal anvendes, når fællesdrevet tilgås.  Du skal fremover bruge den sti, som står i kolonne ”D” i vedlagte."
# Answer: ['There is a change on which path (path) to use when accessing the common drive. In the future, use the path in column "D" of the attached.']
#text = "Jeg vil have et tilbud fra dig. Din ordre kan ikke behandles, fordi den har 14 fejl, der kræver manuel korrektion."
# Answer: ['I want an offer from you. Your order cannot be processed because it has 14 errors that require manual correction.']
#text = "hvordan har du det i dag?"
# Answer: ['How are you today?'] 

if st.button('Translate to English'):
    if text == '':
        st.write('Please enter Danish text for translation') 
    else: 
        da_en_translation_model_name = 'Helsinki-NLP/opus-mt-da-en'
        da_en_model = MarianMTModel.from_pretrained(da_en_translation_model_name)
        da_en_tokenizer = MarianTokenizer.from_pretrained(da_en_translation_model_name)
        da_en_batch = da_en_tokenizer.prepare_seq2seq_batch(src_texts=[text])
        da_en_gen = da_en_model.generate(**da_en_batch)
        da_en_translation = da_en_tokenizer.batch_decode(da_en_gen, skip_special_tokens=True)
        st.write('', str(da_en_translation).strip('][\''))
else: pass


if st.button('Translate to Danish'):
    if text == '':
        st.write('Please enter English text for translation') 
    else: 
        en_da_translation_model_name = 'Helsinki-NLP/opus-mt-en-da'
        en_da_model = MarianMTModel.from_pretrained(en_da_translation_model_name)
        en_da_tokenizer = MarianTokenizer.from_pretrained(en_da_translation_model_name)
        en_da_batch = en_da_tokenizer.prepare_seq2seq_batch(src_texts=[text])
        en_da_gen = en_da_model.generate(**en_da_batch)
        en_da_translation = en_da_tokenizer.batch_decode(en_da_gen, skip_special_tokens=True)
        st.write('', str(en_da_translation).strip('][\''))
else: pass
