# Unified Damage Detection Pro

This upgraded version adds:
- stronger training settings
- saved evaluation metrics
- better Flask responses
- dashboard endpoint
- premium frontend with metric cards, bars, histogram, preview, and raw output

Important honesty:
The code can improve training quality, but it cannot guarantee 85% mAP on weak or imbalanced datasets.
For major gains, simplify classes and improve dataset quality.

Run backend:
cd backend
python -m pip install -r requirements.txt
python app.py

Run frontend:
cd frontend
python -m http.server 5500

Open:
http://127.0.0.1:5500
