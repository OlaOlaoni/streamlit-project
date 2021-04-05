import streamlit as st
from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline

modelname = 'deepset/bert-base-cased-squad2'

model = BertForQuestionAnswering.from_pretrained(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

st.title("Question Answer Demo using BERT")
st.text("")

st.write("Please input Passage here:")
pa = st.text_area("", height=400)

# if not pa:
#     st.warning('Please input a Paragraph.')
#     st.stop()

st.write("Please input Question here:")
que = st.text_input("")

sub_btn = st.button("Submit")

if sub_btn:
    text = nlp({
            'question': que,
            'context': pa
            })
    st.write("Answer is  ", text['answer'])
    st.write("Confidence is ",text['score'])
