import PyPDF2


class ExtractPdf:
    def execute(self, path: str) -> str:
        pdf_file_obj = open(path, 'rb')

        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)

        num_pages = len(pdf_reader.pages)

        text = ""

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        pdf_file_obj.close()

        return text