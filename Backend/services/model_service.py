import os
import sys
import asyncio
from pathlib import Path
from models.schemas import PredictionResponse

class ModelService:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        """
        Register available models with basic metadata.
        Add more models or dynamic loading as needed.
        """
        self.models = {
            "rag_pdf": {
                "description": "RAG QA over a PDF. Provide 'pdf_path' in parameters and use 'text' as the question.",
                "type": "rag",
                "example_parameters": {"pdf_path": "/path/to/doc.pdf", "index_dir": None, "rebuild": False}
            },
            # other model placeholders can be registered here
            "echo": {
                "description": "Simple echo placeholder model for testing.",
                "type": "placeholder"
            }
        }

    def get_available_models(self):
        # Return model metadata dictionary
        return self.models

    async def get_prediction(self, text: str, model_name: str, parameters: dict = None) -> PredictionResponse:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        # Handle RAG over PDF
        if model_name == "rag_pdf":
            pdf_path = None
            index_dir = None
            rebuild = False
            if parameters:
                pdf_path = parameters.get("pdf_path")
                index_dir = parameters.get("index_dir")
                rebuild = bool(parameters.get("rebuild", False))

            if not pdf_path:
                raise ValueError("For 'rag_pdf' model, 'pdf_path' must be provided in parameters.")

            # Ensure project root is on sys.path so RAG.helper can be imported reliably
            project_root = Path(__file__).resolve().parents[2]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            try:
                from RAG.helper import answer_query_from_pdf  # import here to avoid top-level dependency
            except Exception as e:
                raise RuntimeError(f"Failed to import RAG helper: {e}")

            # run the blocking RAG helper in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None,
                answer_query_from_pdf,
                pdf_path,
                text,
                index_dir,
                rebuild
            )

            # You may extend to return retrieved sources/metadata; keep confidence heuristic simple
            return PredictionResponse(
                result=answer,
                confidence=1.0,
                model_name=model_name,
                sources=None
            )

        # Placeholder / other models
        # simple echo placeholder implementation
        result = f"[{model_name} placeholder] {text}"
        confidence = 0.0

        return PredictionResponse(
            result=result,
            confidence=confidence,
            model_name=model_name,
            sources=None
        )
