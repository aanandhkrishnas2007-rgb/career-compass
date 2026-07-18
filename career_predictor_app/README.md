# Career Compass — setup in VS Code

## 1. Project layout
```
career_predictor_app/
  app.py                 <- Flask web server
  train_model.py          <- run once to train + save the model
  requirements.txt
  student_career_dataset.csv   <- ADD YOUR DATASET HERE
  model/                   <- created by train_model.py
  templates/
    index.html
    result.html
  static/
    style.css
    roadmaps/              <- generated roadmap images land here
```

## 2. Install Graphviz (system package, not just the Python library)
- **Windows:** download from https://graphviz.org/download/ and make sure
  "Add Graphviz to PATH" is checked during install.
- **Mac:** `brew install graphviz`
- **Linux:** `sudo apt-get install graphviz`

Without this, `dot.render()` will fail even though `pip install graphviz` succeeded —
the pip package is just a wrapper around the real binary.

## 3. Set up the Python environment (in VS Code's terminal)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

## 4. Add your dataset
Copy `student_career_dataset.csv` into the project root (same folder as `app.py`).

## 5. Train the model (run once, or whenever the CSV changes)
```bash
python train_model.py
```
This creates `model/rf_model.pkl`, `model/label_encoder.pkl`,
`model/model_columns.pkl`, and `model/category_options.pkl`.

## 6. Run the website
```bash
python app.py
```
Open http://127.0.0.1:5000 in your browser.

## Notes on what changed from your Colab script
- Training now happens once in `train_model.py`, not on every prediction.
- `Degree`, `Interest_Area`, and `Internship` are now **dropdowns** built from
  the actual values in your CSV, instead of free-text `input()`. This avoids
  silent mismatches where a typo doesn't match any training category.
- The Graphviz roadmap is rendered to a PNG file in `static/roadmaps/` and
  shown with an `<img>` tag, since Colab's `display()` doesn't work outside
  a notebook.
