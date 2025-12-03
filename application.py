# app.py
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import get_logger

# 1) create the Flask app FIRST
application = Flask(__name__)
app = application

logger = get_logger("flask_app")

# 2) then add routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        # first load of the form
        return render_template("home.html", results=None, error=None)

    # POST -> read form, predict
    try:
        logger.info("Form data: %s", dict(request.form))

        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        df = data.get_data_as_data_frame()
        pipe = PredictPipeline()
        pred = float(pipe.predict(df)[0])
        logger.info("Prediction: %.4f", pred)

        return render_template("home.html", results=pred, error=None)

    except Exception as e:
        logger.exception("Prediction failed")
        return render_template("home.html", results=None, error=str(e))

# 3) run the dev server
if __name__ == "__main__":
    application.run(host="0.0.0.0")
