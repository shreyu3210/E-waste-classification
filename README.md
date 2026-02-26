# E-Waste Classifier

A FastAPI-based application that classifies images into various waste categories (E-waste vs Non-E-waste) using Ultralytics YOLO.

## Project Structure

- `app.py`: FastAPI backend that serves the UI and classification API.
- `predict.py`: Inference utility for CLI and webcam classification.
- `test.py`: Dataset preparation, training and evaluation script.
- `generate_plots.py`: Tool for generating visualizations from training data.
- `TECHNICAL_REPORT.md`: Detailed report on model training, dataset, and performance.
- `templates/`: Contains HTML files used by the UI (e.g. `index.html`).

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create a virtual environment (Optional but Recommended):**

   ```bash
   python -m venv env
   # Activate it (Windows)
   env\Scripts\activate
   # Activate it (Mac/Linux)
   source env/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the model weights:**
   - Make sure your `.pt` model weights (e.g., `best.pt`) are stored in the location expected by the scripts, or update the `MODEL_PATH` variable in `app.py` and `predict.py`.

## Running the App

Start the FastAPI server:

```bash
python app.py
```

Then navigate to [http://localhost:8000](http://localhost:8000) in your browser.

## Using the CLI

You can also use the `predict.py` script for terminal-based predictions:

```bash
python predict.py path/to/image.jpg
python predict.py --webcam
```
