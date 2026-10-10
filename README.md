# Extractive Question Answering with BERT

An extractive question-answering pipeline built on a pre-trained **BERT-large** model fine-tuned on SQuAD. Given a question and a context document, the system finds the exact span of text in the document that answers the question.

Completed as coursework at the University of Sheffield.

---

## How It Works

```
question + document
        │
        ▼
Tokenise with sliding window (384 tokens, stride 192)
        │
        ▼
BERT-large QA model → start & end logits for every token
        │
        ▼
Top-20 start × top-20 end candidates per window
        │
        ▼
Filter invalid spans → pick highest (start + end) score across all windows
        │
        ▼
Clean whitespace → final answer
```

### 1. Model

[`bert-large-uncased-whole-word-masking-finetuned-squad`](https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad), a pre-trained Hugging Face model for extractive QA. It is used as-is for inference, with no further fine-tuning.

### 2. Sliding-window chunking

BERT can only read 512 tokens at once, and documents can be longer. The tokenizer splits each question–document pair into overlapping windows:

| Parameter | Value | Purpose |
|-----------|------:|---------|
| `MAX_LENGTH` | 384 | Maximum tokens per window |
| `DOC_STRIDE` | 192 | Overlap between windows, so answers aren't cut at a boundary |

### 3. Answer span selection

For each window, the model outputs a start score and an end score for every token. Instead of checking every possible pair, the system:

1. Takes the **top 20** start positions and **top 20** end positions (`TOP_K_LOGITS = 20`)
2. Discards spans that:
   - fall outside the document (e.g. inside the question tokens)
   - end before they start
   - are longer than **12 tokens** (`MAX_ANSWER_LENGTH`)
3. Scores each remaining span as `start_logit + end_logit`
4. Returns the highest-scoring span across **all** windows, mapped back to the original text using character offsets

### 4. Output normalisation

Newlines, tabs and repeated spaces in the predicted answer are collapsed into single spaces.

## Results

- **150 questions** answered
- **Total time:** 6.7 s on GPU (CUDA)
- **Average:** ~0.04 s per question (the first question takes longer while the GPU warms up)

Sample predictions:

| Question | Prediction |
|----------|------------|
| What religion was Spielberg's family? | Orthodox Jewish |
| What instrument did Anna Lelkes play? | harp |
| What quantifies the relationship between rest-mass and rest-energy? | E = mc² |
| Who did the Atlantic City Redevelopment Authority partner with in 1999? | Steve Wynn |
| What is on the official Lancashire flag? | red rose on a gold field |

## Error Analysis

Typical failures come from **distractors**: the document contains a plausible answer of the right type (a date, an amount, a name) that relates to a nearby but different part of the text.

| Question | Gold | Wrong prediction | Cause |
|----------|------|------------------|-------|
| In what year was the name *stout* first used to describe a dark beer? | 1677 | 1721 | Confused *stout* with *porter*, which is mentioned in the same passage |
| What was the size of the Notre Dame endowment when Hesburgh became president? | $9 million | $350 million | Picked the endowment value at the *end* of his presidency instead of the start |

**Possible improvements**

- Rank candidate spans partly by how close they are to the key phrase in the question, not just by raw logits
- Add lightweight answer validation to reject nearby distractors, such as later dates or values linked to a different entity

## Repository Structure

```
├── Model.ipynb     # Full pipeline: model loading, inference, export
├── input.json      # 150 questions with their context documents
├── output.json     # Model predictions (generated when you run the notebook)
├── <report>.pdf    # Short report: system description, design choices, error analysis
└── README.md
```

### Input format (`input.json`)

```json
[
  {
    "question_id": "q1",
    "question": "What does the CAR get help with with regards to communication from ITU-D?",
    "document": "Presently, the Central African Republic has active television services..."
  }
]
```

### Output format (`output.json`)

```json
[
  { "question_id": "q1", "answer": "improve infrastructure" }
]
```

## How to Run

Requires Python 3.11. A CUDA GPU is recommended; the code falls back to CPU automatically, but inference will be much slower.

```bash
pip install "numpy<2" pandas
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0
```

Then open `Model.ipynb` and run all cells. The model (~1.3 GB) is downloaded from Hugging Face on the first run, and predictions are saved to `output.json`.

To use your own data, replace `input.json` with a file in the same format.

## Tech Stack

- **PyTorch 2.2** with CUDA 12.1
- **Hugging Face Transformers 4.40**
- **Python 3.11**, Jupyter Notebook

## Author

**Ramiz Aliguliyev**, University of Sheffield
