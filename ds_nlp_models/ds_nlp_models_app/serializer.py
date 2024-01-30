from rest_framework import serializers
from .models import UploadedFile,TranslateModel

class FileUploadSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = UploadedFile
        fields = ('file', 'uploaded_on',)
        
class TranslateModelSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model=TranslateModel
        fields =('file','uploaded_on','language')