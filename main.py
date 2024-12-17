import nltk
from sentence_transformers import SentenceTransformer

from core.CompareWithQuery import CompareWithQuery
from core.DetectLanguage import DetectLanguage
from core.ExtractPdf import ExtractPdf
from core.GetEmbeddings import GetEmbeddings
from core.PreprocessText import PreprocessText
from core.ReadDocuments import ReadDocuments

nltk.download('stopwords')
nltk.download('wordnet')

docs = [
    'SP_CH1.pdf',
    'Problem-Solving Sociology.pdf',
    'EthicsofAI.pdf',
    'Knowledge Graph Embeddings.pdf'
]

def main(query: str):
    pdf_reader = ExtractPdf()
    reader = ReadDocuments(pdf_reader)
    preprocessor = PreprocessText()
    get_embeddings = GetEmbeddings()
    language_detector = DetectLanguage()
    compare_query_with_documents = CompareWithQuery()

    language_detector.execute(query)

    # Leer archivos
    documents = reader.execute(docs)

    # Preprocesas textos
    processed_documents = [preprocessor.execute(doc.content) for doc in documents]
    processed_query = preprocessor.execute(query)

    # Cargar modelo SBERT y generar embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = get_embeddings.execute(model, processed_documents)
    query_embedding = get_embeddings.execute(model, [processed_query])

    # Comparar la consulta con los documentos
    results = compare_query_with_documents.execute(
        query_embedding,
        doc_embeddings,
        documents
    )

    # Mostrar resultados
    print("Resultados de similitud:")
    for index, value in enumerate(results):
        doc, score = value
        print(f"- {doc.name}: {score:.4f}")


# --- Ejecución ---
if __name__ == "__main__":
    main("Knowledge graphs are important in today's society")
