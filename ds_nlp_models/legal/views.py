from django.shortcuts import render
from rest_framework.response import Response 
from django.http import  HttpResponse, JsonResponse
from rest_framework.views import APIView
from rest_framework import status
from rest_framework import parsers
from .service import leagalService
from .serializer import SOWdocumentSerializer,FileUploadSerializer
from .models import GoldContractModel
import os
from rest_framework.parsers import FormParser, MultiPartParser
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample
import json


# Create your views here.

class DocumentCreation(APIView): 
     parser_classes = (parsers.JSONParser,)
     @extend_schema(
        request= SOWdocumentSerializer,
        
    )
     def post(self, request, format=None):
            serializer=SOWdocumentSerializer(data=request.data)
            if serializer.is_valid():
                os.makedirs("./Legal_Output", exist_ok=True)
                path=f'./Legal_Output/dynamic_sow_document_with_tables.docx'
                doc=leagalService.documentCreation(request.data)
                response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
                response['Content-Disposition'] = 'attachment; filename=download.docx'
                doc.save(response)
                leagalService.removeFolder("./Legal_Output")
                return response
            else:
                return Response(serializer.errors,status=status.HTTP_400_BAD_REQUEST) #when serializer is not valid
            
class GoldContract(APIView): 
    queryset = GoldContractModel.objects.all()
    serializer_class=FileUploadSerializer
    parser_classes = (MultiPartParser, FormParser,)
    @extend_schema(
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                     'file': {
                         'type': 'string',
                        'format': 'binary'
                     }
                }
            }
        }
        
    )
    def post(self,request,**kwargs): 
        try:
            serializer = self.serializer_class(data=request.data)
            if serializer.is_valid():
                    file= request.FILES.getlist('file')
                    result=leagalService.goldContract(file)
                    response = HttpResponse(content_type='text/csv')
                    print(type(result))
                    response['Content-Disposition'] = 'attachment; filename=goldCotract.csv'
                    result.to_csv(path_or_buf=response, encoding='utf-8', index=False)
                    return response
                    # respose_dict={'status':status.HTTP_200_OK,'error_msg':""}
                    # return JsonResponse(respose_dict)
                
            else:
                return Response(
                    serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST
                )
        except Exception as e:
            error_msg=str(e)  
            respose_dict={'status':status.HTTP_400_BAD_REQUEST ,'error_msg':error_msg}
            return JsonResponse(respose_dict) 
           
        
    
     