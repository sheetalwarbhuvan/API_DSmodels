import os
import PyPDF2
import yake
# from PIL import Image
import fpdf
import os
      # Directory for storing PDF files
pdf_directory = './content/pdf_files'
    # Directory for storing extracted text from PDFs
text_directory = './content/extracted_text'
def saveFiles(file):
  
    # Create directories if they don't exist
  os.makedirs(text_directory, exist_ok=True)
  os.makedirs(pdf_directory, exist_ok=True)

  with open(os.path.join(pdf_directory, file.name),'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)

def pdfSummary(files):
    fileNames=[]
    for file in files:
        if files is not None:
             saveFiles(file)
             fileNames.append(file.name)
    response=[]
    for file_name in os.listdir(pdf_directory):
        if file_name in fileNames: 
            fileResponse={"fileName":file_name}
            if file_name.endswith('.pdf'):
                # Open the PDF file
                with open(os.path.join(pdf_directory, file_name), 'rb') as file:
                    # Create a PDF reader object
                    reader = PyPDF2.PdfReader(file)

                    # Extract text from each page
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text()

                    # Save the extracted text as a text file
                    text_file_name = file_name.replace('.pdf', '.txt')
                    text_file_path = os.path.join(text_directory, text_file_name)
                    with open(text_file_path, 'w') as text_file:
                        text_file.write(text)
                    fileResponse=keyPhrase(file_name,fileResponse)
                    fileResponse=summary(file_name,fileResponse) 
                    response.append(fileResponse)
        else:
            pass
                 
    return response                  
                    
def keyPhrase(fileName,response):
    if fileName.endswith('.pdf'):
        selected_keys=[]
        # Open the PDF file
        with open(os.path.join(pdf_directory, fileName), 'rb') as file:
              # Create a PDF reader object
              reader = PyPDF2.PdfReader(file)
              # Extract text from each page
              text = ''
              for page in reader.pages:
                  text += page.extract_text()
              kw_extractor = yake.KeywordExtractor(top=3, stopwords=None)
              keywords = kw_extractor.extract_keywords(text)
              selected_keys.append(keywords)
         # Print the generated summaries for each pdf
              for j, keys in enumerate(selected_keys):
                    for kw, v in keys:
                        keyPhrase={"Keyphrase":kw,"Score":v}
                        response['keyphrase']=keyPhrase
    return response

def summary(fileName,response):
    sample_files = []
    pdf_summaries = []  # To store the generated summaries
    with open(os.path.join(pdf_directory, fileName), 'rb') as file:
          # Create a PDF reader object
          reader = PyPDF2.PdfReader(file)

          # Extract text from each page
          text = ''
          for page in reader.pages:
              text += page.extract_text()

          from transformers import T5ForConditionalGeneration,T5Tokenizer

          # Initialize the model and tokenizer
          model = T5ForConditionalGeneration.from_pretrained("t5-base")
          tokenizer = T5Tokenizer.from_pretrained("t5-base")

          # Encode the text
          inputs = tokenizer.encode("summarize: " + text,
          return_tensors="pt", max_length=1000,
          truncation=True)

          # Generate the summary
          outputs = model.generate(inputs,
          max_length=1000, min_length=100,
          length_penalty=2.0, num_beams=4,
          early_stopping=True)

          # Decode the summary
          summary = tokenizer.decode(outputs[0])

          pdf_summaries.append(summary)

        #   pdf.write(5,summary)

        #   pdf.output("Summarise-T.pdf")
    # Print the generated summaries for each pdf
    summaryData=[]
    for i, summary in enumerate(pdf_summaries):
        summarize={"index":i,"summary":summary}
        summaryData.append(summarize)
    response['summary']=summaryData
    return response
    
        
 
            
    
    

  
