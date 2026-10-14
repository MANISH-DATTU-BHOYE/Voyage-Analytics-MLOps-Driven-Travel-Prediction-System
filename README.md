Project Name:-Voyage Analytics: Integrating MLOps in Travel Productionization of ML System

End-to-end Machine Learning + MLOps system for the travel industry — predicting flight prices, classifying users, and recommending hotels, deployed as a production-grade application.

Team: Ashwini Maske · Manish Bhoye · Vedant Deshmukh

📌 Problem Statement

Travel companies sit on huge amounts of underused data — who's flying where, how much they pay, and what hotels they book. This project turns three raw datasets (users, flights, hotels) into a working, production-deployed ML system, while demonstrating real-world MLOps practices (tracking, serving, containerization, orchestration).

💡 Business Insight
Flight pricing is highly predictable from route, agency, and date features — a well-tuned model (R² = 0.99999) can power dynamic pricing displays, fare-alert tools, or budget planning features with near-zero error.
Gender classification from name/company/age alone caps out around 56% accuracy — a useful finding in itself: it shows this data isn't a reliable proxy for demographics, discouraging over-reliance on it for personalization without stronger signals.
Collaborative-filtering hotel recommendations cut search time for travelers by surfacing personalized options from similar users' booking history — a direct lever for conversion and customer satisfaction.
Packaging the whole pipeline with Docker + Kubernetes + MLflow means the business isn't just getting a model — it's getting a system that can be updated, scaled, and monitored like any production software, reducing the cost of maintaining ML in production.
🏗️ Architecture
Users • Flights • Hotels
        │
        ▼
Gender Classifier │ Flight Price Regressor │ Recommender Engine
        │
        ▼
  MLflow → Flask REST API → Docker → Kubernetes + HPA
        │
        ▼
   Streamlit Dashboard → End User
🛠️ Tech Stack

Python scikit-learn Flask Docker Kubernetes MLflow Streamlit

🧩 Modules
#	Module	Approach	Result
1	Flight Price Prediction	Linear Reg. · Decision Tree · Random Forest · Gradient Boosting	Random Forest — R² 0.99999, RMSE 1.14
2	REST API	Flask /predict, /health endpoints	Real-time predictions, e.g. ₹361.96
3	Containerization	Docker image of app + model	Portable, consistent runtime
4	Orchestration	Kubernetes Deployment + NodePort Service	Self-healing, exposed at localhost:5000
5	Autoscaling	HPA on CPU utilization	Scales 1 → 3 replicas at 50% CPU target
6	Experiment Tracking	MLflow (params, metrics, artifacts)	Full experiment comparison & reproducibility
7	Gender Classification	TF-IDF + One-Hot + Logistic Regression	56.34% accuracy, F1 55% (macro)
8	Hotel Recommendation	Collaborative filtering, cosine similarity	Top-5 personalized recommendations/user
9	Dashboard	Streamlit	Interactive UI for all predictions/insights
📊 Model Comparison (Flight Price)
Model	MAE	RMSE	R² Score
Random Forest ✅	0.0925	1.1411	0.99999
Decision Tree	0.0522	2.2671	0.99996
Gradient Boosting	38.1273	50.2538	0.98083
Linear Regression	213.6480	256.6627	0.50007
🚀 Quick Start
bash
# Install
pip install -r requirements.txt

# Run API
python app.py

# Build & run with Docker
docker build -t voyage-analytics .
docker run -p 5000:5000 voyage-analytics

# Deploy to Kubernetes
kubectl apply -f k8s/

# Track experiments
mlflow ui --port 5001

# Launch dashboard
streamlit run streamlit_app.py
📁 Structure
voyage-analytics/
├── data/            # users.csv, flights.csv, hotels.csv
├── models/          # flight_price_model.pkl, gender_classifier.pkl
├── src/             # preprocessing & training scripts
├── app.py           # Flask REST API
├── streamlit_app.py # Dashboard
├── Dockerfile
├── k8s/             # deployment.yaml, service.yaml, hpa.yaml
└── requirements.txt
🔭 Future Scope

Real-time pricing APIs · richer user features · deep-learning recommenders · automated retraining · cloud deployment · centralized monitoring · CI/CD automation

Capstone project demonstrating a complete ML → MLOps production workflow.