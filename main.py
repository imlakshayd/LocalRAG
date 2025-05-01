import os
import pdfplumber # Able to read pdf files and extract text
import magic # Able to get the type of file
import json
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
import chromadb
from chromadb.config import Settings

# Assign directory
directory = r"E:\Winter Sem\LocalRAG\Data"

#Load data file if it already exists
data_file = "data.json"

if os.path.exists(data_file):
    with open(data_file, "r") as file:
        existing_data = json.load(file)
else:
    existing_data = []

print(f"Loaded {len(existing_data)} existing entries.")

# Existing filenames

existing_filenames = set(entry["filename"] for entry in existing_data)

# Iterate over files in directory
new_data = []

for paths, folders, files in os.walk(directory):
    # Open file
    for filename in files:
        if filename in existing_filenames:
            print(f"Skipping already processed file: {filename}")
            continue

        with pdfplumber.open(os.path.join(directory, filename)) as pdf:
            print(f"Procesing file: '{filename}'")
            # Read content of file
            # typeoffile = magic.from_file(filename, mime=True)
            pages = pdf.pages
            for page in pages:
                # Extract text from each page
                text = page.extract_text()

                # print(f"filename: {filename},\n"
                #       f"page: {page},\n"
                #       f"text:\n {text},\n")
                #        f"type: pdf\n")
                # print()

                dictionary = {
                    "filename": f"{filename}\n",
                    "page": f"{page}\n",
                    "text": f"{text}\n",
                    "type": f"pdf\n"
                }
                new_data.append(dictionary)

print(f"Processed {len(new_data)} new entries.")
combined_data = existing_data + new_data

if new_data:
    print(f"Saving total {len(combined_data)} entries to {data_file}")
    with open(data_file, "w") as outfile:
        json.dump(combined_data, outfile, indent=2)
else:
    print("No new files processed. Data file unchanged.")

data_file = "data.json"
json_data = []

with open(data_file, "r") as file:
    json_data = json.load(file)

print(f"Working with {len(json_data)} total items for chunking.")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # Making the max size of characters in one chunk
    chunk_overlap=50 # With each new chunk 50 characters overlap which allows to preserve context
)

all_chunks = []

for entry in json_data:
    full_text = entry["text"]
    filename = entry["filename"]
    page = entry["page"]

    if not full_text: # Skips pages which are empty
        continue

    # Use character based chunking for the text only
    small_chunks = splitter.split_text(full_text)

    for section in small_chunks:
        clean_chunk = section.strip()
        if clean_chunk:
            chunks = (
             {
                "source": f"{filename.strip()} Page {page.strip()}",
                "chunk": clean_chunk
            })

            all_chunks.append(chunks)


print(f"Created {len(chunks)} chunks.")

with open("chunks.json", "w") as outfile:
    json.dump(all_chunks, outfile, indent=2)

with open("chunks.json", "r") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks.")

model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

texts = [chunk["chunk"] for chunk in chunks]

embeddings = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]
        embeddings.append(embedding.squeeze(0).tolist())

print(f"Created {len(embeddings)} embeddings.")

data_to_save = []

for chunk, embedding in zip(chunks, embeddings):
    entry = {
        "source": chunk["source"],
        "chunk": chunk["chunk"],
        "embedding": embedding
    }
    data_to_save.append(entry)

with open("embedded_chunk.json", "w") as outfile:
    json.dump(data_to_save, outfile, indent=2)

print(f"Saved embeddings to embedded_chunks.json.")

client = chromadb.PersistentClient(path="E:/Winter Sem/LocalRAG")

try:
    client.delete_collection(name="info")
    print("Existing collection 'info' deleted.")
except Exception as e:
    print(f"Collection 'info' likely did not exist: {e}")
collection = client.get_or_create_collection(name="info")


all_texts = []
all_embeddings = []
all_metadatas = []
all_ids = []
counter = 0

for chunk, embedding in zip(chunks, embeddings):
    all_texts.append(chunk["chunk"])
    all_embeddings.append(embedding)
    all_metadatas.append({"source": chunk["source"]})
    all_ids.append(str(counter))
    counter += 1

collection.add(
    documents=all_texts,
    embeddings=all_embeddings,
    metadatas=all_metadatas,
    ids=all_ids
)

print("All chunks added to Chroma!")

def embed_query(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]
        return embedding.squeeze(0).tolist()

llm_model_name = "google/gemma-2-2b"


llm_tokenier = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, trust_remote_code=True, device_map="auto")

llm = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenier)



while True:
    question = input("\nAsk a question (or type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    try:
        # Embed the question
        query_embedding = embed_query(question)

        # Search Chroma
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        # Check if results were found
        if not results or not results.get('documents') or not results['documents'][0]:
            print("Could not find relevant context for your question.")
            continue # Ask for a new question

        top_chunks = results['documents'][0]
        context = "\n\n".join(top_chunks)

        system_prompt = f"""
        You are a helpful assistant. Use the provided context to answer the question. Answer concisely based *only* on the context.
        If the context does not contain the answer, say you don't know.

        <context>
        {context}
        </context>

        Question: {question}
        Answer:"""

        print("Generating answer, please wait...")

        # Call the LLM pipeline
        llm_output = llm(system_prompt, max_length=512, do_sample=True, temperature=0.7, truncation=True)


        if not llm_output:
             print("LLM did not return an output.")
             continue


        generated_text = llm_output[0]["generated_text"]

        # Make the marker exactly match the end of your prompt
        prompt_end_marker = f"Question: {question} \nAnswer:" # More precise marker


        # Find it in the *full* generated text, which includes the prompt
        answer_start_index = generated_text.rfind(prompt_end_marker) # Use rfind to find the last occurrence

        if answer_start_index != -1:

            # Get the text *after* the marker
            final_answer = generated_text[answer_start_index + len(prompt_end_marker):].strip()
            print(f"\nAnswer:\n{final_answer}")
        else:

            simple_marker = "Answer:"
            simple_start_index = generated_text.rfind(simple_marker)
            if simple_start_index != -1:
                 final_answer = generated_text[simple_start_index + len(simple_marker):].strip()
                 print(f"\nAnswer (fallback parsing):\n{final_answer}")
            else:
                print("\nRaw Response (could not isolate answer using markers):")
                print(generated_text) # Print the whole thing for debugging


    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        # Optionally add more specific error handling or logging



# while True:
#     question = input("\nAsk a question (or type 'exit' to quit): ")
#
#     if question.lower() == "exit":
#         break
#
#     # Embed the question
#     query_embedding = embed_query(question)
#
#     # Search Chroma
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=3
#     )
#
#     top_chunks = results['documents'][0]
#
#     context = "\n\n".join(top_chunks)
#
#     system_prompt = f"""
#     You are a helpful assistant. Use the provided context to answer the question. Answer concisely based *only* on the context.
#     If the context does not contain the answer, say you don't know.
#
#     <context>
#     {context}
#     </context>
#
#     Question: {question}
#     Answer:"""
#
#     print("Generating answer, please wait...")
#     response = llm(system_prompt, max_length=512, do_sample=True, temperature=0.7, truncation=True)[0]["generated_text"]
#
#     generated_text = response[0]["generated_text"]
#     # Find the position *after* the prompt ends
#     prompt_end_marker = "Answer:"
#     answer_start_index = generated_text.find(prompt_end_marker)
#     if answer_start_index != -1:
#         final_answer = generated_text[answer_start_index + len(prompt_end_marker):].strip()
#         print(f"\nAnswer:\n{final_answer}")
#     else:
#         # Fallback if "Answer:" marker isn't found in the expected place
#         print("\nRaw Response (could not isolate answer):")
#         print(generated_text)
# #
#     # Show Results
# for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
#     print("\n---")
#     print(f"Source: {metadata['source']}")
#     print(f"Content:\n{doc}")


# json_chunks = splitter.split_json(json_data=json_data)
#
# for chunk in json_chunks[:3]:
#     print(chunk)



# for data in json_chunks:
#     with open("data.json", "r") as file:
#         json_data = json.load(file)
#
#     json_chunks = splitter.split_json(json_data=json_data)




