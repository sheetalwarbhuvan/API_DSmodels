from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import fpdf
from PyPDF2 import PdfFileReader
from PyPDF2 import PdfReader
import os


def translateDoc(file,sourceLanguage):
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    language=['Czech','German','French','Estonian','Spanish','Finnish']
    translatedData=""
    if file is not None:
        data=file.read()
        reader=PdfReader(file)
        num_pages=len(reader.pages)
        print(num_pages)
        
        for p in range(num_pages):
            page=reader.pages[p]
            FileContent=page.extract_text()
            text = FileContent.encode('latin-1', 'replace').decode('latin-1')
             #translate Czech to English
            if sourceLanguage=='Czech':
                tokenizer.src_lang = "cs_CZ"
                encoded_cs = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(**encoded_cs,forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
                out=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            #translate French to English
            if sourceLanguage=='French':
                tokenizer.src_lang = "fr_XX"
                encoded_fr = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(**encoded_fr,forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
                out=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
               
            #translate Estonian to English
            if sourceLanguage=='Estonian':
                tokenizer.src_lang = "et_EE"
                encoded_et = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(**encoded_et,forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
                out=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
            #translate German to English
            if sourceLanguage=='German':
                tokenizer.src_lang = "de_DE"
                encoded_de = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(**encoded_de,forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
                out=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
            #translate Spanish to English
            if sourceLanguage=='Spanish':
                tokenizer.src_lang = "es_XX"
                encoded_de = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(**encoded_de,forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
                out=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
               
            #translate Finnish to English
            if sourceLanguage=='Finnish':
                tokenizer.src_lang = "fi_FI"
                encoded_de = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(**encoded_de,forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
                out=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(out)
            translatedData=translatedData + str(out)
           
    else:
        pass
    return translatedData
            # pdf = fpdf.FPDF(format='letter')
            # pdf.add_font('Arial', '', '/content/arial.ttf', uni=True) #path of the font file is set to avoid the error generated by U+2019=','
            # pdf.set_font("Arial", size=12)
            # pdf.add_page()
            # os.makedirs("./TranslateDocuments", exist_ok=True)
            # path=f'./TranslateDocuments/Translated-Source{sourceLanguage}.pdf'
            # pdf.output(path)