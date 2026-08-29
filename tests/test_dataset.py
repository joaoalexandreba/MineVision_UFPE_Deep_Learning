import unittest
import numpy as np
import pandas as pd
import torch
from PIL import Image
import tempfile
from pathlib import Path

from src.preprocessing import HarshDataset, get_transforms
from src.models import HarshNeuralFeatureExtractor


class TestDatasetAndModels(unittest.TestCase):
    """Testes unitários para classes PyTorch Dataset, transformações e redes neurais."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Criar imagens e labels fictícias para teste
        self.img_path = self.temp_path / "test_img.jpg"
        self.lbl_path = self.temp_path / "test_img.txt"

        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img.save(self.img_path)

        with open(self.lbl_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")

        self.df = pd.DataFrame([{
            "image_path": str(self.img_path),
            "label_path": str(self.lbl_path),
            "split_original": "train",
            "classes_presentes": [0, 1],
            "n_classes": 2,
            "strat_key": "0_1",
        }])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_instantiation_and_len(self):
        """Verifica se o HarshDataset é instanciado e tem tamanho correto."""
        dataset = HarshDataset(df=self.df)
        self.assertEqual(len(dataset), 1)

    def test_tensor_transformation_and_shape(self):
        """Verifica se as transformações produzem um Tensor PyTorch no formato (3, 640, 640)."""
        transforms_compose = get_transforms()
        dataset = HarshDataset(df=self.df, transform=transforms_compose)
        tensor_img, label_p = dataset[0]

        self.assertIsInstance(tensor_img, torch.Tensor)
        self.assertEqual(tensor_img.shape, (3, 640, 640))
        self.assertEqual(label_p, str(self.lbl_path))

    def test_neural_feature_extractor_forward(self):
        """Verifica se o módulo PyTorch HarshNeuralFeatureExtractor produz a saída esperada."""
        model = HarshNeuralFeatureExtractor(num_classes=2, in_channels=3)
        dummy_input = torch.randn(4, 3, 640, 640)
        output = model(dummy_input)

        self.assertEqual(output.shape, (4, 2))
        self.assertIsInstance(output, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
