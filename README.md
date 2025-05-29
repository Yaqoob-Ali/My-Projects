# Package Recommendation System

## Project Overview  
The Package Recommendation System is an end-to-end pipeline that forecasts telecom usage metrics (data, SMS, on-net/off-net minutes, social data) for each subscriber’s next billing cycle and then recommends the single best package that minimizes over-provision and cost. It integrates data preprocessing, time-series forecasting, and optimization logic to deliver personalized, actionable package recommendations.

---

## Features

- **Data Preprocessing & Validation**  
  - Cleans raw usage and subscription records  
  - Prorates multi-day package consumption into calendar-month slices  
  - Computes daily usage metrics for each resource  

- **Forecasting Engine**  
  - Uses Prophet on the **last 6 months** of data (with ETS/SARIMA alternatives available)  
  - Falls back to last observed value if history is insufficient  
  - Clamps negative predictions to zero for realistic forecasts  

- **Recommendation Logic**  
  - Normalizes forecasted monthly totals into per-day needs  
  - Filters packages matching the forecasted validity window  
  - Converts package allowances into per-day rates for fair comparison  
  - Selects the package minimizing the **worst over-provision ratio** across all resources  
  - Includes a “closest-diff” fallback when no package fully covers needs  

- **Modular & Extensible Architecture**  
  - Swap out forecasting modules (Prophet, ETS, SARIMA)  
  - Customize recommendation strategies (per-day matching, cost-per-unit)  
  - Clear interfaces for data ingestion and JSON output  

---

## Requirements & Dependencies

- **Python**: 3.8+  
- **Runtime**: Linux / macOS / Windows  
- **Memory**: ≥8 GB RAM (≥16 GB recommended)  
- **Database**: MySQL or PostgreSQL (via SQLAlchemy)  
- **Python Packages** (install via `requirements.txt`):  
  pandas
  sqlalchemy

To install all dependencies at once, run -> "pip install -r requirements.txt"

### Project Engineer:
**Yaqoob Ali** (Associate Data Scientist E&S)

🔗 [LinkedIn](https://linkedin.com/in/yaqoob-ali-data-science-aspirant)
🔗 [Github](https://github.com/Yaqoob-Ali)
