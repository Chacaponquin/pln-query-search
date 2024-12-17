from sentence_transformers import util

from core.Document import Document


# --- Paso 3: Comparación de la consulta con documentos ---
class CompareWithQuery:
    def execute(
        self,
        query_embedding,
        doc_embeddings,
        documents: list[Document]
    ):
        """
        Compara la consulta con los documentos utilizando similitud de coseno.
        """
        # Calcular similitudes
        similarities = util.cos_sim(query_embedding, doc_embeddings)

        # Emparejar documentos con sus similitudes
        results = [(documents[i], float(similarity)) for i, similarity in enumerate(similarities[0])]

        # Ordenar por similitud descendente
        results.sort(key=lambda x: x[1], reverse=True)

        return results