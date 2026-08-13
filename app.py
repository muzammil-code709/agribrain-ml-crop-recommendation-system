import os
import json
from flask import Flask, request, render_template
from src.ml_model import recommender
from src.expert_system import expert_system_insights
from src.utils.logger import logger
from config import MODEL_COMPARISON_PATH

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = os.urandom(24)

@app.route("/")
def index():
    logger.info("Accessed index page.")
    return render_template("index.html", form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    if recommender.load_error:
        logger.error(f"Prediction failed: Models not loaded. {recommender.load_error}")
        return render_template("index.html", error=recommender.load_error, form_data={})

    # Fixed typo: Phosporus -> Phosphorus in form data
    form_data = {
        "Nitrogen": request.form.get("Nitrogen", ""),
        "Phosphorus": request.form.get("Phosphorus", ""),
        "Potassium": request.form.get("Potassium", ""),
        "Temperature": request.form.get("Temperature", ""),
        "Humidity": request.form.get("Humidity", ""),
        "Ph": request.form.get("Ph", ""),
        "Rainfall": request.form.get("Rainfall", ""),
    }

    try:
        N = float(form_data["Nitrogen"])
        P = float(form_data["Phosphorus"])
        K = float(form_data["Potassium"])
        temperature = float(form_data["Temperature"])
        humidity = float(form_data["Humidity"])
        ph = float(form_data["Ph"])
        rainfall = float(form_data["Rainfall"])
        
        # Validation
        if ph < 0 or ph > 14:
            raise ValueError("pH must be between 0 and 14.")
        if any(v < 0 for v in [N, P, K, temperature, humidity, rainfall]):
            raise ValueError("All values must be non-negative.")
            
    except (TypeError, ValueError) as e:
        error_msg = f"Invalid input: {str(e)}" if str(e) else "Please enter valid numerical values for all fields."
        logger.warning(f"Input validation failed: {error_msg}")
        return render_template("index.html", error=error_msg, form_data=form_data)

    values = [N, P, K, temperature, humidity, ph, rainfall]
    logger.info(f"Generating prediction for values: {values}")

    try:
        predictions = recommender.predict(values)
        logger.info(f"Prediction successful. Top crop: {predictions[0]['crop']}")
    except Exception as exc:
        logger.error(f"Error during prediction: {exc}", exc_info=True)
        return render_template("index.html", error=f"Unable to generate prediction: {exc}", form_data=form_data)

    top_crop = predictions[0]["crop"]
    ml_result = f"We highly recommend planting {top_crop.title()} for the provided field conditions."

    # Fetch extended insights
    insights = expert_system_insights(top_crop, values)

    return render_template(
        "index.html",
        predictions=predictions,
        ml_result=ml_result,
        insights=insights,
        form_data=form_data,
    )

@app.route("/metrics")
def metrics():
    """
    Renders the ML Model Comparison Table for evaluation purposes.
    """
    logger.info("Accessed metrics page.")
    metrics_data = []
    try:
        with MODEL_COMPARISON_PATH.open("r") as f:
            metrics_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metrics data: {e}")
        
    return render_template("metrics.html", metrics=metrics_data)

if __name__ == "__main__":
    logger.info("Starting Crop Recommendation App")
    app.run(debug=True)
