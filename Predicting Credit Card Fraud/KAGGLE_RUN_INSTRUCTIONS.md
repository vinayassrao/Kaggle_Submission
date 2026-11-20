# Running v25 (GPT-2 generative classifier) on Kaggle's 2× T4 GPUs

The script is `kaggle_v25_gpt2.py` in this folder. It's self-contained and runs unchanged locally or on Kaggle.

1. **Create the notebook:** on kaggle.com, open the competition https://www.kaggle.com/competitions/playground-series-s6e11 → **Code** → **New Notebook**.
2. **Settings (right-hand panel):** Accelerator → **GPU T4 x2**. **Internet → On** (needed to download GPT-2 the first time). Under **Input**, the competition data is already attached at `/kaggle/input/playground-series-s6e11`.
3. **Add the script:** upload `kaggle_v25_gpt2.py` (Input → Upload → New dataset), or paste its contents into a cell that starts with `%%writefile kaggle_v25_gpt2.py`.
4. **Run, in one cell each:**
   - `!python kaggle_v25_gpt2.py bench --steps 200` (a quick speed check: prints vocabulary size, GPU memory and the time per fold)
   - `!python kaggle_v25_gpt2.py kaggle --epochs 4 --orders 8` (trains folds 0, 2, 4 on GPU 0 and folds 1, 3 on GPU 1 at the same time; each finished fold is saved to `/kaggle/working`, and a rerun skips folds that are already done)
   - `!python kaggle_v25_gpt2.py collect` (builds `v25_gpt2_oof.npy` and `v25_gpt2_test.npy`, and prints the CV AUC)
   Tip: use **Save Version → Save & Run All (commit)** so it keeps running in the background for up to 12 hours.
5. **Bring the results back:** from the notebook's **Output** tab, download `v25_gpt2_oof.npy` and `v25_gpt2_test.npy` (and the `v25_f*_*.npy` parts) into `e:\Kaggle\Predicting Credit Card Fraud\oof\`. Then tell Claude, who will run the stack gate (`v31_stack.py`).
