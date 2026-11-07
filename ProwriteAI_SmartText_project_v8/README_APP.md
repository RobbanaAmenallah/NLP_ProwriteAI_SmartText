# ✨ ProwriteAI — Streamlit demo (`app.py`)

Interactive interface for the Encoder–Decoder models trained from scratch in
`ProwriteAI_SmartText_4Models_Training.ipynb`. Every output is generated live by the saved checkpoints;
no external API and no pretrained model is used.

## Launch
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501.

## What the user sees
* **One text box and one button (✨ IMPROVE MY TEXT).** The user does not choose a model: the app automatically uses
  the checkpoint with the **lowest validation loss** (selection on `validation.csv`, never on the test set). With the
  delivered checkpoints this is `D_Pointer` (BiLSTM + coverage attention + pointer-generator).
* **Confidence indicator** 🟢 / 🟡 / 🔴: the mean log-probability of the generated words. A low score means the model is
  unsure (often a sentence structure it never saw in training), and the app shows a warning so the output is checked.
  The thresholds in `CONFIDENCE_THRESHOLDS` were calibrated on `validation.csv` with `calibrate_confidence.py`.
* Character/word counter, random example, example messages (inputs only), copy-ready output, "Under the hood" panel
  (tokens, IDs, unknown words, copied words, confidence score).
* **⚙️ Advanced settings** (sidebar, collapsed): greedy decoding (default) or beam search (k = 3 / 5).
* **🔬 Academic view** (collapsed section): runs the same message through the four architectures for the project
  presentation; a missing checkpoint shows an error card instead of a result.

## Decoding and post-processing (identical to the notebook)
* Autoregressive greedy decoding or beam search, under `model.eval()` and `torch.inference_mode()`, using only the
  user's message (never a reference text).
* Decoding constraint: a word may not be generated twice in a row.
* `finalize_text`: capitalizes the first letter of each sentence and the pronoun *I*.

## Expected project layout
```
your_project/
├── app.py
├── requirements.txt
├── .streamlit/config.toml          # dark purple theme (optional)
└── prowrite_outputs/               # written by the training notebook
    ├── checkpoints/  A_LSTM.pt  B_BiLSTM.pt  C_Attention.pt  D_Pointer.pt
    ├── artifacts/    vocab.json  tokenizer.json
    └── test_metrics.csv            # optional: scoreboard in the sidebar
```
If your files live elsewhere:
```bash
PROWRITE_CHECKPOINT_DIR=/path/to/checkpoints PROWRITE_ARTIFACT_DIR=/path/to/artifacts \
PROWRITE_OUTPUT_DIR=/path/to/prowrite_outputs streamlit run app.py
```
`PROWRITE_TORCH_THREADS` (default 1 on CPU) sets the number of PyTorch CPU threads.

## Error handling
Clear messages for missing checkpoints or tokenizer files, vocabularies that do not match the checkpoint, invalid model
configurations, empty input and inference exceptions. The app never falls back to echoing the input.

## Recalibrating the confidence thresholds (after retraining)
```bash
PROWRITE_DATA_DIR=data python calibrate_confidence.py 1000
```
and copy the printed thresholds into `CONFIDENCE_THRESHOLDS` in `app.py`.
