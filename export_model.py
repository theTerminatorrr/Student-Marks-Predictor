"""Exports a trained scikit-learn RandomForestRegressor to model.js for the website.

In Colab, add this as the LAST cell of your notebook (after the model is trained):

    from export_model import export_model
    export_model(model, features)

or paste the function below straight into a cell. It downloads model.js for you
and prints the prediction for the sample student, so you can compare it with the site.
"""
import json


def export_model(model, features, path="model.js"):
    assert list(features) == ["study_hours", "attendance_percent", "previous_score", "sleep_hours"], \
        "Feature order must match the website"
    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append({
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "f": [int(f) for f in t.feature],
            "t": [float(v) for v in t.threshold],          # full precision: no rounding
            "v": [float(v[0][0]) for v in t.value],
        })
    with open(path, "w") as f:
        f.write("window.MODEL=" + json.dumps({"trees": trees}, separators=(",", ":")) + ";\n")
    print(f"Saved {path} ({len(trees)} trees)")

    # sample check: the website must give the same number for these inputs
    import pandas as pd
    sample = pd.DataFrame({"study_hours": [7], "attendance_percent": [90],
                           "previous_score": [75], "sleep_hours": [8]})
    print("Sample (7, 90, 75, 8) ->", round(model.predict(sample)[0]))

    try:  # Colab only
        from google.colab import files
        files.download(path)
    except ImportError:
        pass
