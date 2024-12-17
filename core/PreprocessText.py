import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class PreprocessText:
    def execute(self, text: str):
        # Inicializar herramientas de NLTK
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        # 1. Eliminar caracteres especiales y puntuación
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # 2. Convertir a minúsculas
        text = text.lower()

        # 3. Tokenizar, eliminar stopwords y lematizar
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

        # 4. Reconstruir el texto limpio
        return ' '.join(words)