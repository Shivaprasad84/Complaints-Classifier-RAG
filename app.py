import re
import chromadb
import pandas as pd
from rerankers import Reranker
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

client = chromadb.PersistentClient(path="chroma_db")
collection_name = "complaints_collection"
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={
        "hnsw:space": "cosine",
        "hnsw:search_ef": 500
    }
)

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    if collection.count() == 0:
        print("Loading data into the collection...")
        df = pd.read_excel("data/translated_data.xlsx")
        short_description_list = df["Short Description"].astype(str).tolist()
        reported_problem_l1_list = df["Reported Problem Code L1"].astype(str).tolist()
        reported_problem_l2_list = df["Reported Problem Code L2"].astype(str).tolist()
        reported_problem_l3_list = df["Reported Problem Code L3"].astype(str).tolist()

        embeddings = model.encode(short_description_list, batch_size=500,
                                show_progress_bar=True).tolist()
        ids = [str(i) for i in range(len(short_description_list))]
        metadatas = [
            {'L1': l1, 'L2': l2, 'L3': l3} for l1, l2, l3 in zip(
                reported_problem_l1_list,
                reported_problem_l2_list,
                reported_problem_l3_list
            )
        ]

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = short_description_list[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
        print("Data loaded successfully!, Total documents:", collection.count())
    else:
        print("Data already loaded in the collection")

def get_relevant_context(query_text, top_k=5):
    query_emb = model.encode(query_text).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    return '\n\n'.join(
        f"{doc}\n" + '\n'.join([f"{k}: {v}" for k, v in metadata.items()])
        for doc, metadata in zip(docs, metadatas)
    )

def build_rag_pipeline(model_name, context, query):
    prompt_template = """Based solely on the following context, output the corresponding Reported Problem Code L1, Reported Problem Code L2, and Reported Problem Code L3 values.
    IMPORTANT:
    - Do not mix and match the error codes, a given short description will have the same L1, L2 and L3 mappings
    - Just choose one from the following context. Do not change the wordings in L1, L2, L3, return it as it is.
    - The context will always be a set of L1, L2, L3 codes and the question will be based on that context
    - If it is polar opposite of the context, then just answer Non complaint for L1, L2, L3

    Context:
    {context}

    Question:
    {question}

    Output one of the set of L1, L2, L3 codes that is provided as context, choose one which is most suitable set for the given question.
    Note that the set of L1, L2 and L3 will be same for a given short description.

    Output in list format, each in newline like
    - L1: <L1 description>
    - L2: <L2 description>
    - L3: <L3 description>
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model_local = ChatOllama(model=model_name)
    chain = prompt | model_local | StrOutputParser()
    chain_input = {"context": context, "question": query}
    try:
        result = chain.stream(chain_input)
    except Exception as e:
        result = f"Error: {str(e)}"
    return result

def process_message(content):
    pattern = r'<think>(.*?)</think>'
    thinking_parts = re.findall(pattern, content, re.DOTALL)
    message_without_thinking = re.sub(pattern, '', content, flags=re.DOTALL)
    return message_without_thinking.strip(), thinking_parts

def initialize_reranker():
    return Reranker(
        lang="en",
        model_name='cross-encoder',
        model_type='cross-encoder',
        device='cuda'
    )

def rerank(reranker, query_text, context, top_k=5):
    ranked_results = reranker.rank(query=query_text, docs=context).top_k(top_k)
    return '\n\n'.join(f"{result.document.text}" for result in ranked_results)