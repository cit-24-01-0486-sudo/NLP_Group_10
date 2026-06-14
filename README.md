# \# NLP\_Group\_10 — Fake News Detection

# 

# \## CCS3356 Natural Language Processing | Group Assignment 2026

# 

# \---

# 

# \## Group Members

# 

# | Member | Student ID | Name | Role |

# |--------|-----------|------|------|

# | Member 1 | CIT-24-01-0122 | \[Manugi Thennakoon] | Logistic Regression + CNN |

# | Member 2 | CIT-24-01-0486 | \[Tharusha Pieris] | Naïve Bayes + LSTM |

# | Member 3 | CIT-24-01-0598 | \[Chamoda Dulhari] | SVM + DistilBERT |

# 

# \---

# 

# \## Problem Statement

# 

# Fake news has become a critical threat to public trust, democracy, and social stability.

# This project builds an automated fake news detection system using Natural Language

# Processing techniques. Given a news article, the system classifies it as either

# \*\*REAL\*\* or \*\*FAKE\*\* using multiple ML and DL models trained on the ISOT Fake News Dataset.

# 

# Each group member independently develops one ML model and one DL model on the same

# dataset, allowing a structured comparison of approaches ranging from Logistic Regression

# to transformer-based DistilBERT.

# 

# \---

# 

# \## Dataset

# 

# | Detail | Info |

# |--------|------|

# | Name | ISOT Fake News Dataset |

# | Source | https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset |

# | Size | 44,898 articles |

# | Classes | 2 — REAL (0) and FAKE (1) |

# | Files | Fake.csv (23,481 articles) + True.csv (21,417 articles) |

# 

# \---

# 

# \## Models Used

# 

# | Member | ML Model | DL Model |

# |--------|----------|----------|

# | Member 1 | Logistic Regression | CNN (Text) |

# | Member 2 | Naïve Bayes | LSTM |

# | Member 3 | SVM | DistilBERT |

# 

# \---

# 

# \## Project Structure

# 

# ```

# NLP\_Group\_10/

# ├── data/            ← Fake.csv, True.csv, merged.csv

# ├── notebooks/       ← Individual Jupyter notebooks per member

# ├── src/             ← preprocessing.py, utils.py

# ├── models/          ← Saved model files (.pkl, .h5, .pt)

# ├── reports/         ← Final report PDF

# ├── screenshots/     ← App screenshots for report

# ├── videos/          ← Progress video

# ├── app/             ← Streamlit web application (app.py)

# ├── requirements.txt

# └── README.md

# ```

# 

# \---

# 

# \## Setup Instructions

# 

# \### 1. Clone the repository

# ```bash

# git clone https://github.com/cit-24-01-0486-sudo/NLP\_Group\_10.git

# cd NLP\_Group\_10

# ```

# 

# \### 2. Create a virtual environment

# ```bash

# python -m venv venv

# 

# \# Windows

# venv\\Scripts\\activate

# 

# \# Mac/Linux

# source venv/bin/activate

# ```

# 

# \### 3. Install dependencies

# ```bash

# pip install -r requirements.txt

# ```

# 

# \### 4. Download the dataset

# \- Go to https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

# \- Download Fake.csv and True.csv

# \- Place both files inside the `data/` folder

# 

# \### 5. Download GloVe embeddings (for CNN and LSTM models)

# \- Go to https://nlp.stanford.edu/projects/glove/

# \- Download `glove.6B.zip`

# \- Extract and place `glove.6B.100d.txt` inside the `data/` folder

# 

# \---

# 

# \## How to Run the Project

# 

# \### Run individual model notebooks

# Open any notebook from the `notebooks/` folder in Jupyter:

# ```bash

# jupyter notebook

# ```

# 

# \### Run the Streamlit web app

# ```bash

# streamlit run app/app.py

# ```

# Then open your browser at `http://localhost:8501`

# 

# \---

# 

# \## Results Summary

# 

# > Results will be updated after all models are trained and evaluated.

# 

# | Member | Model | Accuracy | F1-Score | ROC-AUC |

# |--------|-------|----------|----------|---------|

# | M1 | Logistic Regression | TBD | TBD | TBD |

# | M1 | CNN | TBD | TBD | TBD |

# | M2 | Naïve Bayes | TBD | TBD | TBD |

# | M2 | LSTM | TBD | TBD | TBD |

# | M3 | SVM | TBD | TBD | TBD |

# | M3 | DistilBERT | TBD | TBD | TBD |

# 

# \---

# 

# \## Branch Structure

# 

# | Branch | Member | Purpose |

# |--------|--------|---------|

# | `main` | All | Final merged code |

# | `feature/CIT-24-01-0122-m1` | Member 1 | LR + CNN development |

# | `feature/CIT-24-01-0486-m2` | Member 2 | NB + LSTM development |

# | `feature/CIT-24-01-0598-m3` | Member 3 | SVM + DistilBERT development |

# 

# \---

# 

# \## Ethics \& Responsible AI

# 

# This system is built for \*\*educational purposes only\*\*.

# 

# \- Predictions should not be used as the sole basis for determining news credibility

# \- The model is trained on 2016–2017 news and may not generalise to current events

# \- Dataset source bias exists — real articles are from Reuters only

# \- Always display confidence scores alongside predictions

# \- Do not log or store user-submitted article text

# 

# \---

# 

# \## Submission Info

# 

# \- \*\*Module:\*\* CCS3356 — Natural Language Processing

# \- \*\*Institution:\*\* Sri Lanka Technology Campus

# \- \*\*Academic Year:\*\* 2026

# \- \*\*Submission:\*\* LMS only



