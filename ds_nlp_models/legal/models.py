from django.db import models


# Create your models here.

class BusinessDetails(models.Model):
    effective_date=models.DateField()
    company_name=models.CharField(max_length=100)
    vendor_name=models.CharField(max_length=100)
    agreement_date=models.DateField()
    businessChallenge=models.CharField(max_length=500)
    project_initiation_fee=models.CharField( max_length=100)
    monthly_service_fee=models.CharField( max_length=100)
    company_representative_name=models.CharField(max_length=100)
    vendor_representative_name=models.CharField(max_length=100)
    company_representative_title=models.CharField(max_length=200)
    vendor_representative_title=models.CharField(max_length=200)
    
  
class GoldContractModel(models.Model):
    file = models.FileField()
