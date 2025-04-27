import os
import pdfplumber # Able to read pdf files and extract text
import magic # Able to get the type of file
import json
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

                with open ("data.json", "w") as outfile:
                    json.dump(new_data, outfile, indent=2)

combined_data = existing_data + new_data


data_file = "data.json"
json_data = []

with open(data_file, "r") as file:
    json_data = json.load(file)


print(f"Loaded {len(json_data)} items")

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

# json_chunks = splitter.split_json(json_data=json_data)
#
# for chunk in json_chunks[:3]:
#     print(chunk)



# for data in json_chunks:
#     with open("data.json", "r") as file:
#         json_data = json.load(file)
#
#     json_chunks = splitter.split_json(json_data=json_data)




