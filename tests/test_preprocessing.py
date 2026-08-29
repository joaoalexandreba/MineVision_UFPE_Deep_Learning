import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocessing import split_dataset


class TestPreprocessing(unittest.TestCase):
    """Testes unitários para divisão estratificada e manipulação de arrays com NumPy."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Criar dataset simulado balanceado com 100 amostras
        registros = []
        for i in range(100):
            strat = "0_1" if i < 60 else "0"
            registros.append({
                "image_path": f"/fake/path/img_{i}.jpg",
                "label_path": f"/fake/path/img_{i}.txt",
                "strat_key": strat,
            })
        self.df = pd.DataFrame(registros)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_split_proportions_80_20(self):
        """Verifica se o split 80/20 divide corretamente o conjunto de 100 amostras."""
        df_train, df_test = split_dataset(
            df=self.df,
            split_dir=self.temp_path,
            train_ratio=0.8,
            seed=42,
        )

        self.assertEqual(len(df_train), 80)
        self.assertEqual(len(df_test), 20)

    def test_split_no_index_overlap(self):
        """Garante que nenhuma amostra de teste vaze para o conjunto de treino."""
        df_train, df_test = split_dataset(
            df=self.df,
            split_dir=self.temp_path,
            train_ratio=0.8,
            seed=42,
        )

        train_imgs = set(df_train["image_path"])
        test_imgs = set(df_test["image_path"])
        overlap = train_imgs.intersection(test_imgs)
        self.assertEqual(len(overlap), 0, "Ocorreu vazamento de dados entre treino e teste!")

    def test_split_files_saved(self):
        """Verifica se os arquivos train.txt e test.txt foram gerados no disco."""
        split_dataset(df=self.df, split_dir=self.temp_path, train_ratio=0.8, seed=42)
        train_file = self.temp_path / "train.txt"
        test_file = self.temp_path / "test.txt"

        self.assertTrue(train_file.exists())
        self.assertTrue(test_file.exists())


if __name__ == "__main__":
    unittest.main()
