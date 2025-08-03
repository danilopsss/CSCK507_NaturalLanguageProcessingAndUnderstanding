### Running FastAPI with uv as Dependency Manager

Follow these steps to run the FastAPI web application using `uv` for dependency management. These instructions are designed for users with no prior experience.

#### 1. Install Python

Make sure you have Python 3.12 or higher installed. You can download it from [python.org](https://www.python.org/downloads/).

To check your Python version, open a terminal and run:
```bash
python --version
```
or
```bash
python3 --version
```

#### 2. Install uv

`uv` is a fast Python dependency manager. Install it using the following command:
```bash
pip install uv
```
If you have multiple Python versions, you may need to use:
```bash
python3 -m pip install uv
```

#### 3. Clone the Project Repository

Download the project files by cloning the repository. Replace `<repo-url>` with the actual repository URL:
```bash
git clone <repo-url>
cd CSCK507_NaturalLanguageProcessingAndUnderstanding
```

#### 4. Install Project Dependencies

Use `uv` to install all required packages listed in `pyproject.toml`:
```bash
uv sync
```
This will automatically read the dependency file and install everything needed.

#### 5. Download spaCy English Language Model

Some features require the spaCy language model. Download it using:
```bash
uv run -- spacy download en_core_web_lg
```

#### 6. Start the FastAPI Web Server

Run the FastAPI application using `uvicorn` (installed as a dependency):
```bash
uv run uvicorn src.main:app --host 0.0.0.0 --reload
```
- `uv run` ensures all dependencies are available.
- `uvicorn` is the server that runs FastAPI.
- `src.chatbot.main:app` points to the FastAPI app instance.
- `--host 0.0.0.0` makes the server accessible on your network.
- `--reload` enables auto-reloading for development.

#### 7. Access the Web Application

Open your browser and go to:
```
http://localhost:8000
```
You should see the chatbot interface.

#### Troubleshooting

- If you see errors about missing dependencies, re-run `uv sync`.
- If `uv` is not recognized, ensure it was installed and your PATH is set correctly.
- For permission errors, try running commands with `sudo` (Linux/macOS) or as administrator (Windows).

#### Running Evaluations

To run model evaluations:

```bash
uv run python evals.py
```

To generate evaluation plots:

```bash
uv run python evals_plotting.py
```

#### Summary

1. Install Python and uv
2. Clone the repository
3. Install dependencies with `uv sync`
4. Download spaCy model
5. Start FastAPI with `uv run uvicorn src.main:app --host 0.0.0.0 --reload`
6. Open `http://localhost:8000` in your browser

You do not need to manually install packages with `pip`—`uv` handles everything for you.
