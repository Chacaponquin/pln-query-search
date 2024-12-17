from core.ExtractPdf import ExtractPdf
from core.Document import Document


class ReadDocuments:
    def __init__(self, pdf_reader: ExtractPdf):
        self.pdf_reader = pdf_reader

    def execute(self, routes: list[str]) -> list[Document]:
        result = []

        for route in routes:
            text = self.pdf_reader.execute(f'docs/{route}')

            document = Document(
                name=route,
                content=text
            )

            result.append(document)

        return result