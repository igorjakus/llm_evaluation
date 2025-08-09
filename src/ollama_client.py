import requests
import json


class OllamaModel:
    def __init__(
        self, base_url="http://localhost:11434/api/generate", model_name="gemma3:1b"
    ):
        self.base_url = base_url
        self.model_name = model_name

    def _stream_response(self, response) -> str:
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                full_response += chunk.get("response", "")
                if chunk.get("done", False):
                    break
        return full_response.strip()

    def generate(self, prompt: str) -> str:
        data = {"model": self.model_name, "prompt": prompt}
        response = requests.post(self.base_url, json=data, stream=True)
        response.raise_for_status()
        return self._stream_response(response)

    @staticmethod
    def model_fn(
        prompt: str,
        base_url: str = "http://localhost:11434/api/generate",
        model_name: str = "gemma3:1b",
    ) -> str:
        return OllamaModel(base_url, model_name).generate(prompt)

    @staticmethod
    def summarize(text: str) -> str:
        prompt = f"Summarize very concisely the following text: {text}"
        return OllamaModel.model_fn(prompt)


if __name__ == "__main__":
    model = OllamaModel()
    output = model.generate("Why is the sky blue?")
    print("Model output:", output)
