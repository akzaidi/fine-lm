---
title: "Improving Multi-lingual Language Understanding Through Contextualized Transfer Learning"
subtitle: "Towards Unsupervised Machine Translation"  
author: "Ali Zaidi"
date: '2018-07-02'
output:
  xaringan::moon_reader:
    lib_dir: libs
    css: xaringan-themer.css
    self_contained: true
    nature:
      highlightStyle: github
      highlightLines: true
      countIncrementalSlides: false
---





class: inverse


## whoami

.pull-left[

![](http://www.angelfire.com/sd/kreelah/images/STUPENDOUSMAN.JPG)

]

.pull-right[

* Ali Zaidi
* ![](https://github.com/carlsednaoui/gitsocial/raw/master/assets/icons%20without%20padding/github.png) akzaidi, ![](https://github.com/carlsednaoui/gitsocial/raw/master/assets/icons%20without%20padding/twitter.png) alikzaidi
* From: DC, Ghana, Kyrgyzstan, Pakistan, Toronto, NYC, Bay Area
* Work: Microsoft (prior: Revolution Analytics, NERA Economic Consulting)
* Student: Stanford (prior: University of Toronto)
* Interests: Food, Music, Stochastic Analysis, Languages, NLP, Reinforcement Learning

]

--

* Interests for the camp: meeting cool people and learning about you, your favorite music and food, and maybe a few phrases from you native languages!
* Transfer learning in NLP
* Multi-task learning in NLP
* Structured prediction models: semantic indexing, relation extraction
* Grounded language learning in Minecraft

---

background-image: url("https://media.giphy.com/media/1M9fmo1WAFVK0/giphy.gif")
background-size: cover

---

## Data Scarcity, Non-generalizeable Features


Dataset | Domain | Size (#sentences)
---------|----------|---------
 CoNLL 2003 | NER | 15K
 OntoNotes | Coreference Resolution | 75K
 PTB | Parsing | 40K
 SQuAD | QA | 100K
 SNLI | Entailment | 570K
 SST | Sentiment | 10K

- Most SOTA NLP results are obtained by training end-to-end architectures for each language task. 
- Most datasets are in English, and are very small.
- I'd like examine how transformer models and convolutional modules such as QANet can be used as generative language models
- Can we fine-tune these language models for text classification and multi-task problems? 
- Is the combination of transformer/convolutional architectures + structured prediction models effective representations for multi-lingual transfer learning?

---

## Recent Approaches

### Fine-Tuning Language Models

1. [OpenAI: Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/)
    * **_tldr_**: Train an unsupervised language model using a transformer architecture, and then fine-tune on task-specific datasets.
2. [fastAI: Universal Language Model Fine-tuning for Text Classification](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
    * **_tldr_**: Pre-train a language model on large unlabelled corpus. Initialize new language model on your unlabeled domain-specific corpus. Fine-tune task-domain-specific architecture for text classification.
    ![](imgs/ulmfit.png)

---

## Recent Approaches

### Comprehension Models and Multi-task Networks

3. [Google Brain: QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension](https://arxiv.org/pdf/1804.09541.pdf)
    - **_tldr_**: Transformer based Q&A model consisting solely of convolutions and self-attentions.
4. [AllenAI: Deep Contextualized Word Vectors](https://arxiv.org/abs/1802.05365)
    - **_tldr_**: Train a generic language model using Bidirectional-LSTM and iteratively fine-tune contextual vectors.
5. [Salesforce Research, The Natural Language Decathlon](https://einstein.ai/research/the-natural-language-decathlon)
    - **_tldr_**: Challenge consisting of ten NLP tasks. Proposed MQAN: bidirectional LSTM, dual coattention, + additional two BiLSTMs + self-attention.
    
    
---

![](https://einstein.ai/static/images/pages/research/decaNLP/MQAN.png)
    
---


class: center
background-image: url(https://media.giphy.com/media/falTqLTgSgmas/giphy.gif)
background-size: cover

# Thanks!

[`https://github.com/akzaidi/fine-lm`](https://github.com/akzaidi/fine-lm)
