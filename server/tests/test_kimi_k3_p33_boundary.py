import tempfile
import unittest
from pathlib import Path

from scripts.analyze_kimi_k3_p33_boundary import parse_boundary


class P33BoundaryProfileTest(unittest.TestCase):
    def test_requires_exactly_92_routed_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stderr.log"
            rows = [
                '[kimi-k3-boundary] phase="routed layer preparation" '
                f'compute_ms={layer}.0 total_ms={layer}.1'
                for layer in range(92)
            ]
            rows.append("[kimi-k3-stage] position=7 tokens=1")
            path.write_text("\n".join(rows) + "\n")
            parsed = parse_boundary(path)
            self.assertEqual(parsed, [(7, [float(i) for i in range(92)])])

            path.write_text("\n".join(rows[:-2] + rows[-1:]) + "\n")
            with self.assertRaises(ValueError):
                parse_boundary(path)


if __name__ == "__main__":
    unittest.main()
