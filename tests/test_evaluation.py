import unittest
import numpy as np
from src.evaluation import calculate_confidence_interval


class TestEvaluation(unittest.TestCase):
    """Testes unitários para cálculos de métricas estatísticas e intervalos de confiança."""

    def test_confidence_interval_calculation(self):
        """Verifica o cálculo de média, desvio padrão e limites do IC 95% via t-Student."""
        values = np.array([0.98, 0.99, 0.97, 0.96, 1.00])
        media, desvio, ic_inf, ic_sup, t_crit = calculate_confidence_interval(values, confidence=0.95)

        self.assertAlmostEqual(media, 0.98, places=2)
        self.assertGreater(desvio, 0.0)
        self.assertLess(ic_inf, media)
        self.assertGreater(ic_sup, media)
        self.assertGreater(t_crit, 2.0)

    def test_confidence_interval_zero_variance(self):
        """Verifica comportamento do IC quando todos os valores são idênticos."""
        values = np.array([0.95, 0.95, 0.95])
        media, desvio, ic_inf, ic_sup, _ = calculate_confidence_interval(values, confidence=0.95)

        self.assertAlmostEqual(media, 0.95, places=5)
        self.assertAlmostEqual(desvio, 0.0, places=5)
        self.assertAlmostEqual(ic_inf, 0.95, places=5)
        self.assertAlmostEqual(ic_sup, 0.95, places=5)


if __name__ == "__main__":
    unittest.main()
