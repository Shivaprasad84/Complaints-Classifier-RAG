import nest_asyncio
nest_asyncio.apply()

import torch
# Workaround for torch error in Streamlit
torch.classes.__path__ = []

import streamlit as st
from models import available_models
from app import \
    get_relevant_context, build_rag_pipeline, process_message, initialize_reranker, rerank, load_data

st.set_page_config(page_title="Complaints Classifier", layout="wide", page_icon=":speech_balloon:")

st.sidebar.header("Settings")

if "available_models" not in st.session_state:
    st.session_state["available_models"] = available_models

selected_model = st.sidebar.selectbox("Choose your model", st.session_state["available_models"])
use_reranker = st.sidebar.checkbox("Use Reranker", value=True)


if st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = []

if st.sidebar.button("Populate DB"):
    load_data()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "model" not in st.session_state:
    st.session_state["model"] = selected_model
else:
    st.session_state["model"] = selected_model

if "reranker" not in st.session_state and use_reranker:
    st.session_state["use_reranker"] = True
    st.session_state["reranker"] = initialize_reranker()
elif not use_reranker:
    st.session_state["use_reranker"] = False

st.title("Complaints Classifier Chat")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            processed, think_parts = process_message(message["content"])
            st.markdown(processed)
            for think in think_parts:
                with st.expander(f"Show model thinking"):
                    st.markdown(think)
        else:
            st.markdown(message["content"])

user_query = st.chat_input("Enter your query")
if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            context = get_relevant_context(user_query, top_k=10)
            if st.session_state["use_reranker"]:
                context_chunks = context.split('\n\n')
                final_context = rerank(st.session_state["reranker"], user_query, context_chunks)
            else:
                final_context = '\n\n'.join(context.split('\n\n')[:5])
            print(final_context)
            print('---'*50)
            results_generator = build_rag_pipeline(st.session_state["model"], final_context, user_query)

            placeholder = st.empty()
            full_response = ""
            thinking_parts = []

            for chunk in results_generator:
                full_response += chunk
                processed, think_parts = process_message(full_response)
                placeholder.markdown(processed)
                thinking_parts = think_parts

            for i, think in enumerate(thinking_parts):
                with st.expander(f"Show model thinking {i+1}"):
                    st.markdown(think)

        st.session_state["messages"].append({"role": "assistant", "content": full_response})