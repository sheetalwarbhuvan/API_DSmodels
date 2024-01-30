from django.shortcuts import render
from rest_framework.response import Response 
from django.http import  JsonResponse
from rest_framework.views import APIView
from rest_framework import status
from .serializer import FileUploadSerializer,TranslateModelSerializer
from .models import UploadedFile,TranslateModel
from rest_framework.parsers import FormParser, MultiPartParser
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import _stop_words
import os
import PyPDF2
import fpdf
from .service import churnPrediction


import re


def cleanExtractedData(text_file):
    clean_cont=[]
    with open(text_file) as f:
        clean_cont = f.read().splitlines()
    shear=[i.replace('\xe2\x80\x9c','') for i in clean_cont ]
    shear=[i.replace('\xe2\x80\x9d','') for i in shear ]
    shear=[i.replace('\xe2\x80\x99s','') for i in shear ]
    shears = [x for x in shear if x != ' ']
    shearss = [x for x in shears if x != '']
    dubby=[re.sub("[^a-zA-Z]+", " ", s) for s in shearss]
    print(dubby)
    return dubby

def filePreprocessing(file):
     # Directory for storing PDF files
    pdf_directory = './content/pdf_files'
    # Directory for storing extracted text from PDFs
    text_directory = './content/extracted_text'
    # Create directories if they don't exist
    os.makedirs(text_directory, exist_ok=True)
    os.makedirs(pdf_directory, exist_ok=True)
    with open(os.path.join(pdf_directory, file.name),'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)
   # for fileObj in file:
    #if file is not None:
    #         save_uploadedfile(file)
    file_name=file.name
    print(file_name)
    for file_name in os.listdir(pdf_directory):
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
                return cleanExtractedData(text_file_path)
    
def topic_modelling(dubby):
    vect=CountVectorizer(ngram_range=(1,1),stop_words='english')
    dtm=vect.fit_transform(dubby)
    pd.DataFrame(dtm.toarray(),columns=vect.get_feature_names_out())
    lda=LatentDirichletAllocation(n_components=5)
    lda.fit_transform(dtm)
    lda_dtf=lda.fit_transform(dtm)
    sorting=np.argsort(lda.components_)[:,::-1]
    features=np.array(vect.get_feature_names_out())
    # mglearn.tools.print_topics(topics=range(5), feature_names=features,
    # sorting=sorting, topics_per_chunk=5, n_words=10)
    topic_res=[]
    for topic_idx, topic in enumerate(lda.components_):
        top_word_indices = topic.argsort()[:-10 - 1:-1]  # Top 10 words per topic
        top_words = features[top_word_indices]
        print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")
        topic_res.append(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")
    summary=summerize(lda_dtf,dubby)
    print(topic_res)
    return topic_res,summary
    
    
def summerize(lda_dtf,dubby):
    summerizeTxt=''
    #Sorting documents based on the third topic in descending order
    Agreement_Topic=np.argsort(lda_dtf[:,2])[::-1]

    # Printing the first four documents with their first two sentences
    for i in Agreement_Topic[:4]:  #This selects the first four indices from the sorted list.
        print(".".join(dubby[i].split(".")[:2]) + ".\n") #the loop then iterates over these indices (i) and prints the first two sentences of each document. It does this by splitting the text in each document (dubby[i]) by periods (.),
                                                        #taking the first two elements, and joining them back with a period.
    Domain_Name_Topic=np.argsort(lda_dtf[:,4])[::-1]
    for i in Domain_Name_Topic[:4]:
        summerizeTxt+=".".join(dubby[i].split(".")[:2]) + "."
        print(summerizeTxt)
    return summerizeTxt

class SummerizeModel(APIView):
    queryset = UploadedFile.objects.all()
    serializer_class = FileUploadSerializer
    parser_classes = (MultiPartParser, FormParser,)
  
    def post(self,request):
        try:
            serializer = self.serializer_class(data=request.data)
            if serializer.is_valid():
                # uploaded_file = serializer.validated_data["file"]
                # serializer.save()
                dubby=filePreprocessing(request.FILES['file'])
                topicRes,summary=topic_modelling(dubby)
                
                #return the payload 
                status = True
                error_msg = ""
                respose_dict={'status':status,'error_msg':error_msg,'topics':topicRes,'summary':summary}
                return JsonResponse(respose_dict)
            
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            status=False  
            error_msg=str(e)  
            summary=""
            topicRes=[]
            respose_dict={'status':status,'error_msg':error_msg,'topics':topicRes,'summary':summary}
            return JsonResponse(respose_dict) 

class ChurnPredictionModel(APIView):
    queryset = UploadedFile.objects.all()
    serializer_class = FileUploadSerializer
    parser_classes = (MultiPartParser, FormParser,)
    def post(self,request):
        try:
            serializer = self.serializer_class(data=request.data)
            if serializer.is_valid():
                uploaded_file = request.FILES['file']
                churnPrediction.dataPreprocessing(uploaded_file)
                status = True
                error_msg = ""
                respose_dict={'status':status,'error_msg':error_msg,'response':'Chur prediction completed'}
                return JsonResponse(respose_dict);
        except Exception as e:
            status=False  
            error_msg=str(e) 
            print("Error :")
            print(e) 
            summary=""
            topicRes=[]
            respose_dict={'status':status,'error_msg':error_msg}
            return JsonResponse(respose_dict) 
    
   



# Create your views here.
