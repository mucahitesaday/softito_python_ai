"""Data drift ve sezgisel optimizasyon örneklerinin birim testleri."""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path: Path, name: str):
    """Sayısal klasör adlarından bağımsız biçimde örnek modülü yükler."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PaketTesti(unittest.TestCase):
    """İki ders paketinin temel davranışlarını doğrular."""

    @classmethod
    def setUpClass(cls):
        cls.drift = load(
            ROOT / "Monitoring" / "data_drift_monitor.py",
            "drift",
        )
        cls.optimization = load(
            ROOT / "Optimization" / "evolutionary_optimizers.py",
            "optimization",
        )

    def test_drift_raporu(self):
        reference = self.drift.veri_uret(600, 0, 1)
        production = self.drift.veri_uret(600, 0.5, 2)
        report = self.drift.drift_raporu(reference, production)

        self.assertEqual(len(report), 8)
        self.assertGreater(report["drift"].sum(), 2)

    def test_domain_classifier(self):
        reference = self.drift.veri_uret(500, 0, 1)
        production = self.drift.veri_uret(500, 0.6, 2)
        result = self.drift.domain_classifier(reference, production)

        self.assertGreater(result["roc_auc"], 0.7)

    def test_genetic_algorithm(self):
        _, best_value, history = self.optimization.genetic_algorithm(
            dim=3,
            generations=80,
        )
        self.assertLess(best_value, 15)
        self.assertEqual(len(history), 80)

    def test_cma_es(self):
        _, best_value, history = self.optimization.cma_es(
            dim=3,
            generations=60,
        )
        self.assertGreaterEqual(best_value, 0)
        self.assertEqual(len(history), 60)

    def test_particle_swarm(self):
        _, best_value, history = self.optimization.particle_swarm(
            dim=3,
            iterations=90,
        )
        self.assertLess(best_value, 2)
        self.assertEqual(len(history), 90)


if __name__ == "__main__":
    unittest.main()
