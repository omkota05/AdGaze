# AdGaze

## Setup

1. Create and activate a virtualenv, then install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install torch opencv-python matplotlib fastapi uvicorn
   ```

2. `DeepGaze/` is gitignored (it's a separate cloned repo, not part of this codebase). Clone and install it manually:

   ```bash
   git clone https://github.com/matthias-k/DeepGaze.git
   cd DeepGaze
   pip install -e .
   ```

3. Run the backend:

   ```bash
   uvicorn backend.main:app --reload
   ```

   Then check `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`.
