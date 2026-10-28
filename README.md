# MarkPredict

Predicts a student's exam score from study hours, attendance, previous score and sleep.
The Random Forest runs in the browser from `model.js`, so the site is fully static.

**Files:** `index.html`, `style.css`, `script.js`, `model.js` (the website) · `student_mark_predict.py`, `export_model.py`, `student_performance_messy_expanded.csv` (training).

## Run locally
```
python -m http.server
```
Open http://localhost:8000. Double-clicking `index.html` works too.

## Retrain
Run `student_mark_predict.py` (Colab works; upload `export_model.py` and the CSV first). It writes a new `model.js`; replace the old one. Sample check: inputs 7 / 90 / 75 / 8 give 93 on both the script and the site.

## Deploy
- **Vercel:** import the repo, framework preset "Other", no build command, empty output directory.
- **GitHub Pages:** Settings > Pages > Deploy from branch > `main` / root.
