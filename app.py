# App to generate Blogs using Llama2 Blog GGML model on the local
# developed by Meta AI that is based on the original LLAMA-2. 
# It has been converted to F32 and quantized to 4 bits, which makes it more efficient in terms of memory and computational requirements. 
# The model can be used for a variety of natural language understanding and generation tasks, such as: Text completion, Text generation, Conversation modeling, and Semantic similarity estimation. 


# GGML is a tensor library
# CTransformers is a python bind for GGML.

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from Llama2 model
def get_Llama_response(input_text, no_of_words, blog_style):
    
    # Call Llama2 model from local using CTransformer
    llm = CTransformers(
        model="models\llama-2-7b-chat.ggmlv3.q5_1.bin",
        model_type='llama',
        local_files_only = True,
        config={'max_new_tokens':256,
                'temperature':0.01})
    
    # Prompt Template
    
    template="""
        Write a blog for {blog_style} job profile on the topic {input_text} 
        within {no_of_words} words.    
    """
    
    prompt=PromptTemplate(
        input_variables=["blog_style","input_text","no_of_words"],
        template=template
    )
    
    # Generate response from the Llama2 model
    
    chain = prompt | llm
    response = chain.invoke({"blog_style":blog_style, "input_text":input_text, "no_of_words":no_of_words})
    print(response)
    return response
   
    

st.set_page_config(
    page_title="Blog Generation",
    page_icon='🤖🦙',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 🤖🦙")

input_text=st.text_input("Enter the Blog Topic")

# Creating 2 more fields

col1,col2=st.columns([5,5])

with col1:
    no_of_words=st.text_input('No. of Words')
    
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers', 'Data Scientists', 'Common People'),index=0)
    
submit=st.button("Generate Blog") 

# Final Response
if submit:
    st.write(get_Llama_response(input_text, no_of_words, blog_style))