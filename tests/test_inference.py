import unittest
from pathlib import Path
from src.inference import predict_image


class TestInference(unittest.TestCase):
    """Testes unitários para o módulo de inferência."""

    def test_inference_missing_image_raises_error(self):
        """Garante que imagem inexistente lance FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            predict_image(
                image_path="/caminho/falso/foto_nao_existe.jpg",
                model_weights="yolo11n-seg.pt",
            )

    def test_inference_missing_weights_raises_error(self):
        """Garante que arquivo de pesos inexistente lance FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            predict_image(
                image_path=Path(__file__),
                model_weights="/caminho/falso/pesos_nao_existem.pt",
            )


if __name__ == "__main__":
    unittest.main()
