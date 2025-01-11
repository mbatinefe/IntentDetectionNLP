# Intent Classification and Slot Filling

This project implements several approaches for multi-intent classification and slot filling. It includes multiple models and comparison of their performance.

## Models Implemented
- Logistic Regression with TF-IDF 
- Naive Bayes
- BERT
- Word2Vec with Logistic Regression
- TF-IDF with SVM
- Custom Slot-Gated Model (BiLSTM + Attention)

## Dataset Structure
- Train (70%): 24,498 samples
- Dev (15%): 5,250 samples
- Test (15%): 5,250 samples

### Intents Distribution
- AddToPlaylist
- BookRestaurant
- GetWeather
- PlayMusic
- RateBook
- SearchCreativeWork
- SearchScreeningEvent

## Data Preprocessing
- Text lowercase conversion
- Special character removal
- Stopword removal
- Lemmatization using SpaCy
- TF-IDF vectorization
- Word2Vec embeddings

## Performance Metrics
- Intent Classification Accuracy
- Slot F1 Score
- Semantic Accuracy (both intent and slots correct)

## Results

### a) TF-IDF and Logistic Regression
Transforms text data into numerical vectors, highlighting the importance of words in the corpus. Logistic
Regression achieved high F1 scores for frequent intents like “AddToPlaylist” but struggled with more
complex or nuanced queries. Example classification report on test data:

- AddToPlaylist: Precision 0.99, Recall 1.00, F1-Score 0.99  
- BookRestaurant: Precision 0.99, Recall 0.98, F1-Score 0.98  
- GetWeather: Precision 1.00, Recall 0.93, F1-Score 0.96  
- …

### b) Naive Bayes
Naive Bayes was implemented on TF-IDF features, offering speedy classification with slightly lower
accuracy on rarer classes. For instance, F1 scores decreased for “GetWeather” and “PlayMusic.” However,
it remains a viable option for quick baseline tasks.

### c) BERT
Leverages Transformer-based architecture for contextual understanding. It excelled in multi-intent classification,
outperforming other models in more complex categories like “SearchCreativeWork.” Achieved near-perfect
micro F1-scores (~0.98), at the cost of higher computational overhead.

### d) Word2Vec
Trains embeddings on the combined text, aggregates sentence vectors, then applies logistic regression for
multi-intent classification. This approach performed well on frequent classes but had moderate difficulty
with rarer or context-dependent intents.

### e) TF-IDF and SVM
Combines TF-IDF vectorization with a OneVsRest SVM approach. Achieved substantial accuracy across
all intents—most notably excelling in frequent categories like “AddToPlaylist.”

### f) Base Model
Originally derived from a paper that used SNIPS rather than MixSNIPS. Adapting the code required extensive modifications and updates due to outdated TensorFlow and data preprocessing steps. We ultimately wrote our own code based on the mathematical foundations from the paper to handle multi-intent detection effectively.

<img width="447" alt="Screenshot 2025-01-11 at 8 39 59 PM" src="https://github.com/user-attachments/assets/acaffabf-7f48-40e2-8385-af854572ef70" />

## Citation
@inproceedings{van-der-goot-etal-2021-masked,
    title = "From Masked Language Modeling to Translation: Non-{E}nglish Auxiliary Tasks Improve Zero-shot Spoken Language Understanding",
    author = {van der Goot, Rob  and
      Sharaf, Ibrahim  and
      Imankulova, Aizhan  and
      {\"U}st{\"u}n, Ahmet  and
      Stepanovi{\'c}, Marija  and
      Ramponi, Alan  and
      Khairunnisa, Siti Oryza  and
      Komachi, Mamoru  and
      Plank, Barbara},
    editor = "Toutanova, Kristina  and
      Rumshisky, Anna  and
      Zettlemoyer, Luke  and
      Hakkani-Tur, Dilek  and
      Beltagy, Iz  and
      Bethard, Steven  and
      Cotterell, Ryan  and
      Chakraborty, Tanmoy  and
      Zhou, Yichao",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.197",
    doi = "10.18653/v1/2021.naacl-main.197",
    pages = "2479--2497",
}
