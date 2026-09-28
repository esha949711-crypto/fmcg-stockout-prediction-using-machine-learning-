from pathlib import Path
import json
import joblib
import pandas as pd

from flask import Flask, render_template, request, send_from_directory
from jinja2 import ChoiceLoader, FileSystemLoader
from analytics_data import AnalyticsError, read_analytics_data, summarize

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
# Support both the checked-in flat layout and a conventional templates folder.
app.jinja_loader = ChoiceLoader([
    FileSystemLoader(str(ROOT / "templates")),
    FileSystemLoader(str(ROOT)),
])
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def existing_path(*names):
    return next((ROOT / name for name in names if (ROOT / name).is_file()), ROOT / names[0])


MODEL_PATH = existing_path("models/best_model.joblib", "best_model.joblib")
META_PATH = existing_path("models/model_metadata.json", "model_metadata.json")
DATA_PATH = existing_path("data/inventory_data.csv", "data/processed/processed_inventory_data.csv", "processed_inventory_data.csv")
RESULTS_PATH = existing_path("outputs/results/model_comparison.csv", "model_comparison.csv")
FIGURE_PATH = ROOT / "outputs" / "figures"
PREDICTION_PATH = ROOT / "outputs" / "predictions.csv"
PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
FEATURES = ["current_stock","sales_velocity","lead_time_days","demand_variability","promotion_status","price","discount","category","region","weather_condition","seasonality","day_of_week","month"]

# Analytics remains usable even if a saved model is unavailable or incompatible.
try:
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
except Exception:
    app.logger.exception("Unable to load prediction model")
    model = None

meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {
    "best_model": "Not trained", "risk_thresholds": {"medium": 0.35, "high": 0.65}
}
results = pd.read_csv(RESULTS_PATH).to_dict("records") if RESULTS_PATH.exists() else []


@app.route("/")
def index():
    return render_template("index.html", meta=meta)


@app.route(
    "/predict",
    methods=["GET","POST"]
)

def predict():

    prediction = None

    error = None

    if request.method == "POST":

        try:

            data = {

                "current_stock":
                float(request.form["current_stock"]),

                "sales_velocity":
                float(request.form["sales_velocity"]),

                "lead_time_days":
                float(request.form["lead_time_days"]),

                "demand_variability":
                float(request.form["demand_variability"]),

                "promotion_status":
                int(request.form["promotion_status"]),

                "price":
                float(request.form["price"]),

                "discount":
                float(request.form["discount"]),

                "category":
                request.form["category"],

                "region":
                request.form["region"],

                "weather_condition":
                request.form["weather_condition"],

                "seasonality":
                request.form["seasonality"],

                "day_of_week":
                int(request.form["day_of_week"]),

                "month":
                int(request.form["month"])

            }

            if model is None:

                raise Exception(
                    "Trained model not found."
                )

            row = pd.DataFrame(
                [data],
                columns=FEATURES
            )

            probability = float(

                model.predict_proba(row)[0,1]

            )

            medium = float(
                meta["risk_thresholds"]["medium"]
            )

            high = float(
                meta["risk_thresholds"]["high"]
            )

            if probability >= high:

                risk="High Risk"

            elif probability >= medium:

                risk="Medium Risk"

            else:

                risk="Low Risk"

            label = (

                "YES"

                if probability >= 0.50

                else "NO"

            )

            action = {

                "High Risk":
                "Review replenishment immediately.",

                "Medium Risk":
                "Monitor inventory and upcoming demand.",

                "Low Risk":
                "Continue normal inventory monitoring."

            }[risk]

            prediction = {

                "label":label,

                "probability": probability,

                "probability_percent":
                round(probability*100,2),

                "risk":risk,

                "action":action,

                "inputs":data

            }

            # SAVE PREDICTION HISTORY

            save = pd.DataFrame([{

                **data,

                "probability":probability,

                "risk":risk,

                "prediction":label

            }])

            save.to_csv(

                PREDICTION_PATH,

                mode="a",

                header=not PREDICTION_PATH.exists(),

                index=False

            )

        except Exception as e:

            error=str(e)

    return render_template(

        "predict.html",

        prediction=prediction,

        error=error

    )


@app.route("/analytics", methods=["GET", "POST"])
def analytics():
    source = request.form.get("source", "upload") if request.method == "POST" else request.args.get("source", "dataset")
    context = {"source": source, "source_name": "", "stats": [], "charts": [], "notes": [], "error": None}
    status = 200
    try:
        if source == "upload" and request.method == "POST":
            upload = request.files.get("file")
            if upload is None or not upload.filename:
                raise AnalyticsError("Choose a CSV file to analyse.")
            if not upload.filename.lower().endswith(".csv"):
                raise AnalyticsError("Choose a file with a .csv extension.")
            data = read_analytics_data(upload.stream)
            context["source_name"] = "Uploaded CSV: " + upload.filename
        elif source in ("dataset", "predictions"):
            path = DATA_PATH if source == "dataset" else PREDICTION_PATH
            context["source_name"] = path.name
            if not path.exists():
                if source == "predictions":
                    raise AnalyticsError("No saved predictions yet. Make a prediction first, then refresh this view.")
                raise AnalyticsError("Dataset not found. Add processed_inventory_data.csv to the project folder or upload a CSV below.")
            data = read_analytics_data(path)
        else:
            raise AnalyticsError("Choose Dataset or Saved predictions, or upload a CSV.")
        context.update(summarize(data, meta.get("risk_thresholds")))
    except AnalyticsError as exc:
        context["error"] = str(exc)
        status = 400
    response = app.make_response((render_template("analytics.html", **context), status))
    # Re-read current data on every request and prevent cached dashboard results.
    response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(413)
def upload_too_large(error):
    return render_template("analytics.html", source="upload", source_name="",
                           stats=[], charts=[], notes=[],
                           error="CSV is too large. Maximum upload size is 16 MB."), 413


@app.route("/static/style.css")
def stylesheet():
    directory = ROOT / "static" if (ROOT / "static" / "style.css").is_file() else ROOT
    return send_from_directory(directory, "style.css")


@app.route("/analytics.css")
def analytics_stylesheet():
    return send_from_directory(ROOT, "analytics.css")


@app.route("/models")
def models():
    return render_template("models.html", results=results, meta=meta)


@app.route("/figures/<path:filename>")
def figures_file(filename):
    return send_from_directory(FIGURE_PATH, filename)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
