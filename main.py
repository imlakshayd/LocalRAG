import os
import pdfplumber # Able to read pdf files and extract text
import magic # Able to get the type of file
import json
import requests
from langchain_text_splitters import RecursiveJsonSplitter

# Assign directory
directory = r"E:\Winter Sem\LocalRAG\Data"

# Iterate over files in directory

for paths, folders, files in os.walk(directory):
    # Open file
    for filename in files:
        with pdfplumber.open(os.path.join(directory, filename)) as pdf:
            print(f"Content of '{filename}'")
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

                with open ("data.json", "a") as outfile:
                    json.dump(dictionary, outfile)
                    outfile.write("\n")


json_data = requests.get("data.json").json()
splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = splitter.split_json(json_data = json_data )

