# Clinical RAG Pipeline — Parkinson's & Alzheimer's Literature

A Retrieval-Augmented Generation (RAG) system for clinical question-answering over neurodegenerative disease research literature. Built as a demonstration of applied clinical NLP skills for doctoral research in clinical data science.

---

## Motivation

Clinical research teams working with large volumes of biomedical literature face significant challenges in retrieving disease-specific evidence efficiently. Manually searching and synthesising hundreds of abstracts is time-consuming and prone to gaps.

This pipeline demonstrates how RAG architectures can support **evidence-based clinical question answering** — directly relevant to clinical data science research in neurodegenerative diseases such as Parkinson's and Alzheimer's. It is designed with scalability in mind: the same architecture applies to patient-level EHR text, discharge summaries, and structured clinical databases.

---

## Architecture

```
PubMed API (Entrez)
        ↓
Abstract collection & cleaning (Biopython, Pandas)
        ↓
Text chunking (LangChain RecursiveCharacterTextSplitter)
chunk_size=400, chunk_overlap=50
        ↓
Clinical embedding (Bio_ClinicalBERT)
emilyalsentzer/Bio_ClinicalBERT
        ↓
Vector storage & retrieval (ChromaDB, cosine similarity)
        ↓
Grounded answer generation (Groq — LLaMA 3.1 8B)
        ↓
Cited, source-grounded clinical answer
```

---

## Dataset

- **Source:** PubMed via NCBI Entrez API (no proprietary data, fully reproducible)
- **Queries used for corpus construction:**
  - `Parkinson's disease machine learning biomarkers`
  - `Parkinson's disease electronic health records NLP`
  - `Parkinson's disease multimodal data deep learning`
  - `Alzheimer's disease clinical data harmonisation OMOP`
- **Corpus size:** 500+ unique abstracts after deduplication by PMID
- **Metadata retained:** PMID, title, publication year, source query
- **No proprietary or patient-level data used** — fully open and reproducible

---

## Key Design Decisions

### Embedding model: Bio_ClinicalBERT
`emilyalsentzer/Bio_ClinicalBERT` was chosen over general-purpose sentence transformers (e.g. `all-MiniLM-L6-v2`) because it was pre-trained on MIMIC-III clinical notes. This gives it substantially better semantic alignment with medical terminology, clinical abbreviations, and disease-specific language — critical for meaningful retrieval over biomedical text.

### Chunking strategy
`RecursiveCharacterTextSplitter` with `chunk_size=400` and `chunk_overlap=50` tokens. The overlap is deliberately set to preserve clinical context across sentence boundaries — a sentence describing a biomarker finding may be semantically incomplete without the preceding sentence establishing the patient cohort.

### Vector store: ChromaDB with cosine similarity
Cosine similarity is used as the distance metric rather than Euclidean distance, as it is invariant to embedding magnitude — more appropriate for comparing normalised sentence-level representations.

### LLM: LLaMA 3.1 8B via Groq (temperature=0.1)
Temperature is set to 0.1 to prioritise factual grounding over generative creativity. The system prompt instructs the model to answer only from provided context and to cite PMIDs — reducing hallucination risk in clinical contexts.

### Retrieval: top-5 chunks per query
Evaluated at k=1, k=3, and k=5. Top-5 balances context richness with prompt length constraints.

---

## Evaluation Queries

Five clinically targeted queries aligned with LCSB's research focus on neurodegenerative disease data infrastructure:

1. What machine learning methods have been used to predict Parkinson's disease progression?
2. How have NLP techniques been applied to clinical notes for Parkinson's diagnosis?
3. What biomarkers are used for early detection of Parkinson's disease?
4. How is multimodal data integrated for Alzheimer's disease research?
5. What are the challenges of harmonising clinical data across hospitals for neurodegenerative disease research?

---

## Retrieval Evaluation

Retrieval quality assessed using average cosine similarity at k=1, 3, and 5 across all test queries.

| Query (truncated) | Avg similarity@1 | Avg similarity@3 | Avg similarity@5 |
|---|---|---|---|
| ML methods for Parkinson's progression... | — | — | — |
| NLP on clinical notes for Parkinson's... | — | — | — |
| Biomarkers for early Parkinson's detection... | — | — | — |
| Multimodal data for Alzheimer's... | — | — | — |
| Clinical data harmonisation challenges... | — | — | — |

> *Fill in your similarity scores from the evaluation cell output before pushing to GitHub.*

A query semantic similarity heatmap is included (`query_similarity_heatmap.png`) showing the degree of semantic overlap between evaluation queries — useful for identifying redundancy in query design.

---

## Visualisations

| File | Description |
|---|---|
| `abstracts_by_year.png` | Distribution of corpus abstracts by publication year |
| `query_similarity_heatmap.png` | Cosine similarity matrix across evaluation queries using Bio_ClinicalBERT embeddings |

---

## Repository Structure

```
clinical-rag-parkinsons/
├── clinical_rag_parkinsons.ipynb   # Main notebook — full pipeline end to end
├── parkinsons_abstracts.csv        # Fetched and cleaned PubMed corpus
├── abstracts_by_year.png           # Corpus year distribution plot
├── query_similarity_heatmap.png    # Query embedding similarity heatmap
├── retrieval_evaluation.csv        # Retrieval quality scores at k=1,3,5
├── requirements.txt                # Python dependencies
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/clinical-rag-parkinsons.git
cd clinical-rag-parkinsons
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
Get a free key at [console.groq.com](https://console.groq.com). Then either:
- Set as environment variable: `export GROQ_API_KEY=your_key_here`
- Or in Colab: add via Runtime > Secrets > `GROQ_API_KEY`

### 4. Run the notebook
Open `clinical_rag_parkinsons.ipynb` in Jupyter or Google Colab and run cells in order.

---

## Requirements

```
biopython
chromadb
sentence-transformers
langchain-text-splitters
langchain-community
groq
pandas
numpy
matplotlib
seaborn
```

---

## Limitations & Next Steps

- **Data scope:** Currently limited to PubMed abstracts. The next step is extending to **MIMIC-III discharge summaries** (PhysioNet access in progress) for patient-level unstructured clinical text — moving from literature retrieval to patient-level clinical question answering.
- **Structured data integration:** Planned integration with **OMOP CDM**-mapped structured variables alongside unstructured text, enabling hybrid retrieval across both modalities.
- **Federated retrieval:** Future direction — extending the ChromaDB retrieval layer to operate across distributed document stores in a privacy-preserving manner, aligned with federated learning principles for multi-site clinical data.
- **Evaluation:** Similarity-based retrieval evaluation is a proxy metric. Ground-truth QA evaluation (e.g. using BioASQ benchmarks) is a planned next step.
- **Embedding model:** Bio_ClinicalBERT is an older BERT-based model. Replacing with `MedCPT` or `BioMedBERT-large` would likely improve retrieval quality for longer queries.

---

## Relevance to Doctoral Research

This project was built as a practical demonstration of skills directly relevant to doctoral research in clinical data science — specifically the integration of NLP pipelines, semantic retrieval, and LLM-based generation over biomedical text. The architecture is designed to generalise to the kinds of clinical data environments described in LCSB's work: multimodal, heterogeneous, and requiring FAIR-compliant, privacy-aware processing.

---

## Author

**Shweta Debjit Sarkar**
Independent researcher in clinical machine learning
[GitHub](https://github.com/Shweta-Portfolio) · [LinkedIn](https://www.linkedin.com/in/shwetapooja/) · [Email](mailto:shweta.sarkar.academic@gmail.com)
