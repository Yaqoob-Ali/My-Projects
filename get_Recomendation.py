from flask import Flask, request, jsonify
from datetime import datetime

# Import the model logic
from Model_T3 import get_package_recommendation

app = Flask(__name__)

def error_response(message, code=400):
    return jsonify({"error": message}), code

@app.route("/recommend-usage", methods=["POST"])
def recommend_usage():
    """
    Expects JSON payload:
      {
        "msisdn": 3000000010,
        "forecast_date": "YYYY-MM-DD"
      }
    Returns JSON with:
      {
        "forecast": { … },
        "recommendation": { … },
        "generated_at": "…"
      }
    """
    data = request.get_json(silent=True)
    if not data:
        return error_response("Request must be valid JSON", 415)

    msisdn = data.get("msisdn")
    forecast_date_str = data.get("forecast_date")

    if not msisdn:
        return error_response("Missing 'msisdn' parameter", 400)
    if not forecast_date_str:
        return error_response("Missing 'forecast_date' parameter", 400)

    try:
        forecast_date = datetime.strptime(forecast_date_str, "%Y-%m-%d")
    except ValueError:
        return error_response("Invalid 'forecast_date' format, expected YYYY-MM-DD", 400)

    try:
        # Call the central recommendation function
        response = get_package_recommendation(msisdn, forecast_date)
        return jsonify(response), 200
    except ValueError as ve:
        # e.g. "No data for this MSISDN" or "Invalid MSISDN format"
        return error_response(str(ve), 404)
    except Exception as e:
        return error_response(f"Internal server error: {e}", 500)

@app.route("/", methods=["GET"])
def index():
    return (
        "<h2>MSISDN Package Recommendation API</h2>"
        "<p>POST JSON to <code>/recommend-usage</code> with <code>msisdn</code> and <code>forecast_date</code>.</p>"
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
