import json
import pandas as pd
import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('chat_list.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()
st.header('부산소프트웨어마이스터고 챗봇')
st.subheader("안녕하세요 소마고 챗봇입니다.")
html = '''
    <div class='
    
    
    >
    
    
    
    '''


with st.form('form', clear_on_submit=True):
    user_input = st.text_input('사용자 : ', '')
    submitted = st.form_submit_button('전송')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    if answer['distance'] > 0.5:
        st.session_state.generated.append(answer['답변'])
    else:
        st.session_state.generated.append('적절한 답변이 없습니다. 정확한 답변을 듣고 싶으시다면 051-971-2153으로 연락주세요.')

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    st.markdown(
    html.format(st.session_state['past'][i], st.session_state['generated'][i])
    , unsafe_allow_html=True)
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
        

tab1, tab2, tab3 = st.tabs(["학교소개", "입학안내", "문의"])
with tab1:
    st.header("소개")
with tab2:
    st.header("입학 안내")
with tab3:
    st.header("물어보실")
    


    