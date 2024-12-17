from sentence_transformers import SentenceTransformer
from torch import Tensor


# --- Paso 2: Representación con SBERT ---
class GetEmbeddings:
    """
    Genera embeddings para una lista de textos utilizando SBERT.
    """
    def execute(self, model: SentenceTransformer, texts: list[str]) -> Tensor:
        return model.encode(texts, convert_to_tensor=True)