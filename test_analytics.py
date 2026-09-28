import io
from pathlib import Path
from contextlib import contextmanager
from uuid import uuid4
import unittest
from unittest.mock import Mock, patch
import numpy as np

from analytics_data import AnalyticsError, read_analytics_data, summarize


CSV = """current_stock,sales_velocity,demand_variability,stockout_within_7_days,category,region,promotion_status
10,2,1,0,Food,North,0
30,6,3,1,Food,South,1
50,4,2,1,Home,North,1
"""


@contextmanager
def temporary_directory():
    # Keep fixtures inside the checkout, also usable in restricted runners.
    directory = Path(__file__).parent / (".analytics-test-" + uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        for file in directory.iterdir():
            file.unlink()
        directory.rmdir()


class CalculationTests(unittest.TestCase):
    def test_known_totals_and_group_rates(self):
        result = summarize(read_analytics_data(io.StringIO(CSV)))
        stats = {s["label"]: s["value"] for s in result["stats"]}
        self.assertEqual(stats["Records analysed"], "3")
        self.assertEqual(stats["Average current stock"], "30.00")
        self.assertEqual(stats["Derived stockout rate"], "66.67%")
        self.assertEqual(stats["Derived stockout-positive records"], "2")
        charts = {c["title"]: c for c in result["charts"]}
        rows = charts["Derived stockout rate by category"]["rows"]
        self.assertEqual({r["label"]: r["value"] for r in rows}, {"Food": 50, "Home": 100})
        for title in ("Current stock distribution", "Sales velocity distribution", "Demand variability distribution"):
            self.assertEqual(sum(r["value"] for r in charts[title]["rows"]), 3)

    def test_predictions_are_not_treated_as_actual_stockouts(self):
        frame = read_analytics_data(io.StringIO("current_stock,probability\n10,0.1\n20,0.35\n30,0.65\n"))
        result = summarize(frame)
        self.assertNotIn("Derived stockout rate", [s["label"] for s in result["stats"]])
        risk = next(c for c in result["charts"] if c["title"] == "Predicted risk bands")
        self.assertEqual([r["value"] for r in risk["rows"]], [1, 1, 1])

    def test_invalid_csv_has_clear_error(self):
        for csv in ("", "current_stock\n", "other\n1\n", "current_stock\nwrong\n",
                    "current_stock\n-1\n", "current_stock\ninf\n",
                    "current_stock,probability\n1,1.2\n",
                    "current_stock,stockout_within_7_days\n1,2\n"):
            with self.subTest(csv=csv), self.assertRaises(AnalyticsError):
                read_analytics_data(io.StringIO(csv))

    def test_single_class_and_constant_column(self):
        result = summarize(read_analytics_data(io.StringIO("current_stock,stockout_within_7_days\n0,0\n0,0\n")))
        self.assertEqual(result["charts"][0]["rows"][0]["value"], 2)
        self.assertEqual([r["value"] for r in result["charts"][1]["rows"]], [2, 0])


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        cls.module = app
        cls.client = app.app.test_client()

    def test_dataset_changes_are_visible_without_restart(self):
        with temporary_directory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text(CSV)
            with patch.object(self.module, "DATA_PATH", path):
                first = self.client.get("/analytics")
                self.assertEqual(first.status_code, 200)
                self.assertIn(b"66.67%", first.data)
                self.assertNotIn(b"<img", first.data)
                path.write_text("current_stock,stockout_within_7_days\n100,0\n")
                second = self.client.get("/analytics")
                self.assertIn(b"100.00", second.data)
                self.assertNotEqual(first.data, second.data)
                self.assertEqual(second.headers["Cache-Control"], "no-store")

    def test_upload_isolated_from_saved_data(self):
        response = self.client.post("/analytics", data={"source": "upload", "file": (io.BytesIO(CSV.encode()), "new.csv")})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Uploaded CSV: new.csv", response.data)
        self.assertIn(b"66.67%", response.data)
        invalid = self.client.post("/analytics", data={"file": (io.BytesIO(b"wrong\n1"), "bad.csv")})
        self.assertEqual(invalid.status_code, 400)
        self.assertIn(b"Missing current_stock", invalid.data)
        self.assertNotIn(b"analytics-stats", invalid.data)

    def test_missing_predictions_and_invalid_source(self):
        with patch.object(self.module, "PREDICTION_PATH", Path("does-not-exist.csv")):
            response = self.client.get("/analytics?source=predictions")
            self.assertEqual(response.status_code, 400)
            self.assertIn(b"No saved predictions yet", response.data)
        self.assertEqual(self.client.get("/analytics?source=unknown").status_code, 400)

    def test_flat_repository_pages_and_styles(self):
        for url in ("/", "/predict", "/models", "/about", "/static/style.css", "/analytics.css", "/analytics"):
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                response.close()
        with self.client.get("/static/style.css") as response:
            self.assertIn(b"--ink", response.data)

    def test_prediction_history_refreshes(self):
        with temporary_directory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text("current_stock,probability\n100,0.2\n")
            with patch.object(self.module, "PREDICTION_PATH", path):
                first = self.client.get("/analytics?source=predictions")
                with path.open("a") as handle:
                    handle.write("10,0.8\n")
                second = self.client.get("/analytics?source=predictions")
                self.assertIn(b"50.00%", second.data)
                self.assertNotEqual(first.data, second.data)
                self.assertNotIn(b"Derived stockout rate</span>", second.data)

    def test_prediction_submission_reaches_analytics(self):
        form = dict(current_stock="20", sales_velocity="8", lead_time_days="3",
                    demand_variability="2", promotion_status="0", price="5",
                    discount="0", category="Groceries", region="North",
                    weather_condition="Sunny", seasonality="Summer",
                    day_of_week="1", month="6")
        model = Mock()
        model.predict_proba.return_value = np.array([[0.2, 0.8]])
        with temporary_directory() as directory:
            path = Path(directory) / "predictions.csv"
            with patch.object(self.module, "PREDICTION_PATH", path), patch.object(self.module, "model", model):
                response = self.client.post("/predict", data=form)
                self.assertEqual(response.status_code, 200)
                data = read_analytics_data(path)
                self.assertEqual(len(data), 1)
                self.assertAlmostEqual(data.probability.iloc[0], 0.8)
                dashboard = self.client.get("/analytics?source=predictions")
                self.assertIn(b"80.00%", dashboard.data)
                self.assertIn(b"High risk", dashboard.data)

    def test_oversized_upload_and_escaped_labels(self):
        with patch.dict(self.module.app.config, MAX_CONTENT_LENGTH=100):
            response = self.client.post("/analytics", data={"file": (io.BytesIO(b"x" * 101), "large.csv")})
            self.assertEqual(response.status_code, 413)
        data = b'current_stock,category,stockout_within_7_days\n1,<script>alert(1)</script>,1\n'
        response = self.client.post("/analytics", data={"file": (io.BytesIO(data), "data.csv")})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"<script>", response.data)
        self.assertIn(b"&lt;script&gt;", response.data)


if __name__ == "__main__":
    unittest.main()
