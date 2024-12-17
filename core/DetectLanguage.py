from langdetect import detect


class DetectLanguage:
    def execute(self, text: str) -> str:
        language = detect(text)
        return language
