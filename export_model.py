"""Exports a trained scikit-learn RandomForestRegressor to model.json for the website.

Usage: add these two lines at the END of student_mark_predict.py, then run it:

    from export_model import export_model
    export_model(model, features)

Copy the generated model.json next to index.html.
Feature order must match the website: study_hours, attendance_percent, previous_score, sleep_hours.
"""
import json


def export_model(model, features, path="model.json"):
    assert features == ["study_hours", "attendance_percent", "previous_score", "sleep_hours"]
    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append({
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "f": [int(f) for f in t.feature],
            "t": [round(float(v), 4) for v in t.threshold],
            "v": [round(float(v[0][0]), 2) for v in t.value],
        })
    with open(path, "w") as f:
        json.dump({"trees": trees}, f, separators=(",", ":"))
    print(f"Saved {path} ({len(trees)} trees)")
