# GEMFormer
Global Explicit Memory Transformer
## Citation
We highly appreciate your act of staring and citing. Your attention to detail and recognition is greatly valued.
```
@inproceedings{sagirova-burtsev-2023-uncertainty,
    title = "Uncertainty Guided Global Memory Improves Multi-Hop Question Answering",
    author = "Sagirova, Alsu  and
      Burtsev, Mikhail",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.262",
    doi = "10.18653/v1/2023.emnlp-main.262",
    pages = "4317--4328",
    abstract = "Transformers have become the gold standard for many natural language processing tasks and, in particular, for multi-hop question answering (MHQA). This task includes processing a long document and reasoning over the multiple parts of it. The landscape of MHQA approaches can be classified into two primary categories. The first group focuses on extracting supporting evidence, thereby constraining the QA model{'}s context to predicted facts. Conversely, the second group relies on the attention mechanism of the long input encoding model to facilitate multi-hop reasoning. However, attention-based token representations lack explicit global contextual information to connect reasoning steps. To address these issues, we propose GEMFormer, a two-stage method that first collects relevant information over the entire document to the memory and then combines it with local context to solve the task. Our experimental results show that fine-tuning a pre-trained model with memory-augmented input, including the most certain global elements, improves the model{'}s performance on three MHQA datasets compared to the baseline. We also found that the global explicit memory contains information from supporting facts required for the correct answer.",
}
```
