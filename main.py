import os
import pdfplumber # Able to read pdf files and extract text
import magic # Able to get the type of file

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
                print(f"filename: {filename},\n"
                      f"page: {page},\n"
                      f"text:\n {text},\n")
                      # f"type: {typeoffile}")
                print()