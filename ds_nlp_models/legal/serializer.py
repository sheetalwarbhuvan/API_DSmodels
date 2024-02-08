from rest_framework import serializers
from .models import BusinessDetails,GoldContractModel
class SOWdocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model=BusinessDetails
        fields='__all__'
        
class FileUploadSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = GoldContractModel
        fields = ('file',)
   