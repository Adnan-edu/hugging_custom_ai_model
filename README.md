# PyTorch Deep Learning - Text Classification with Hugging Face

# **Table of Contents**

1. [Overview](#overview)

2. [What is Hugging Face?](#what-is-hugging-face)

3. [What We’re Going to Build](#what-were-going-to-build)

4. [What is Text Classification?](#what-is-text-classification)

5. [Why Train Your Own Text Classification Models?](#why-train-your-own-text-classification-models)

6. [Workflow We’re Going to Follow](#workflow-were-going-to-follow)

7. [Importing Necessary Libraries](#importing-necessary-libraries)

8. [Getting a Dataset](#getting-a-dataset)

9. [Food Not Food Image Caption Dataset Creation](#food-not-food-image-caption-dataset-creation)

10. [Where Can You Get More Datasets?](#where-can-you-get-more-datasets)

11. [Setting Up a Model for Training](#setting-up-a-model-for-training)

12. [Acknowledgments](#acknowledgments)


## Overview

The text outlines a process for text classification that involves starting with a text dataset, building a classification model, and creating a shareable demo. The workflow will utilize various open-source tools from the Hugging Face ecosystem.

<img src="https://camo.githubusercontent.com/f2b03524e937ea49501331f8ae479cf59fb7c119ffd194125bd9f326ac043bc7/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f6d7264626f75726b652f6c6561726e2d68662d696d616765732f7265736f6c76652f6d61696e2f6c6561726e2d68662d746578742d636c617373696669636174696f6e2f30302d70726f6a6563742d666f6f642d6e6f742d666f6f642d6f766572766965772e706e67"
width="800">

## What is Hugging Face?

- **Hugging Face Overview**  
  - Hugging Face is a leading AI company specializing in **Natural Language Processing (NLP)** and **Machine Learning (ML)**.  
  - It provides **pretrained models, datasets, and tools** for deep learning applications.  
  - Hugging Face has become the go-to platform for **open-source AI research** and deployment.  

- **Key Components**  
  - **[Transformers Library](https://huggingface.co/docs/transformers/)** – Provides state-of-the-art NLP models like **BERT, GPT, T5, DistilBERT** for tasks such as text classification, summarization, and translation.  
  - **[Datasets Library](https://huggingface.co/docs/datasets/)** – Access and process large-scale ML datasets efficiently.  
  - **[Tokenizers Library](https://huggingface.co/docs/tokenizers/)** – Fast, optimized tokenization for various transformer models.  
  - **[Accelerate Library](https://huggingface.co/docs/accelerate/)** – Simplifies distributed training and multi-GPU setups.  
  - **[Evaluate Library](https://huggingface.co/docs/evaluate/)** – Provides evaluation metrics for ML models.  

- **Hugging Face Hub**  
  - **[Model Hub](https://huggingface.co/models)** – Hosts thousands of pretrained ML models.  
  - **[Dataset Hub](https://huggingface.co/datasets)** – Collection of ready-to-use datasets.  
  - **[Spaces](https://huggingface.co/spaces)** – Platform for hosting and sharing AI-powered demos with **Gradio** and **Streamlit**.  

- **Why Use Hugging Face?**  
  - **Pretrained models** allow for **transfer learning** to improve performance with minimal data.  
  - **Easy-to-use APIs** integrate with **PyTorch, TensorFlow, and JAX**.  
  - Supports **open-source collaboration** and community-driven AI advancements.  


## What we're going to build

• Building a text classification model to identify whether text (like image captions) is about food or not food
• This is similar to technology used in the Nutrify app that helps users learn about food
• The project follows three main steps:

1. **Data**: Problem definition and dataset preparation
2. **Model**: Finding, training and evaluating a text classification model using Hugging Face
3. **Demo**: Creating and sharing a demo for others to use
   • The finished project will result in a trained model with a shareable demo hosted on Hugging Face

## What is text classification?

• Text classification is the process of assigning categories to text (words, phrases, sentences, paragraphs, or documents)

• Common text classification problems include spam detection, sentiment analysis, language detection, topic classification, hate speech detection, and product categorization

• Classification types include binary (one thing or another), multi-class (one from many), and multi-label (one or more from many)

• Text classification is widely used in business settings, such as categorizing insurance claims

• Models for text classification include:

- Rule-based (simple but requires manual rule creation)
- Bag of Words (simple but doesn't capture word order)
- TF-IDF (weighs word importance but doesn't capture word order)
- Deep learning (can learn complex patterns but requires more data/compute)

• Deep learning models often perform better with quality datasets, and Hugging Face facilitates their implementation

An example text classification problem to classify insurance claim texts into at fault or not fault. This result of the model would send the claim to a different department in the insurance company.

<img src="https://camo.githubusercontent.com/eab65c3cbc7258857d03819185fa0ae702b9ee494261d512bb31413ee5a2651f/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f6d7264626f75726b652f6c6561726e2d68662d696d616765732f7265736f6c76652f6d61696e2f6c6561726e2d68662d746578742d636c617373696669636174696f6e2f30322d6578616d706c652d746578742d636c617373696669636174696f6e2d776f726b666c6f772e706e67"
width="800">


## Why train your own text classification models?

Based on the document about why to train your own text classification models, here are the most important points:

• You can use pre-trained models, API-powered models, or large language models (GPT, Gemini, Claude, Mistral) for text classification

• Training/fine-tuning your own model offers several advantages:

- Full control over model lifecycle
- No usage limits (except compute constraints)
- Train once and deploy anywhere
- Better privacy with in-house data handling
- Often faster performance with small, specialized models

• Using pre-built model APIs has different benefits:

- Easy setup with minimal code
- No need to maintain compute resources
- Access to advanced models
- Ability to scale with usage

• API models come with drawbacks:

- Dependency on third-party service uptime
- Data must be sent externally
- May have daily/periodic usage limits
- Often slower due to API call requirements

For this project, we're going to focus on fine-tuning our own model.

## Workflow we're going to follow

- **Workflow Overview**: The process follows the motto _"data, model, demo!"_ for structured machine learning development.
- **Data Preparation**:
  - Create and preprocess the dataset for training.
- **Model Definition**:
  - Utilize `transformers.AutoModelForSequenceClassification` to define a text classification model.
- **Hyperparameter Configuration**:
  - Define training arguments using `transformers.TrainingArguments` (controls optimization, scheduling, etc.).
- **Training Setup**:
  - Initialize a `transformers.Trainer` instance with the training arguments and dataset.
- **Model Training**:
  - Execute training with `Trainer.train()`.
- **Model Saving**:
  - Save the trained model locally or push it to the Hugging Face Hub.
- **Evaluation & Testing**:
  - Generate predictions on test data and analyze model performance.
- **Deployment & Demo**:
  - Convert the trained model into a shareable demo.
- **Non-Linearity of ML Projects**:

  - The workflow provides structure but allows flexibility for iterative experimentation.
 

But this worfklow will give us some good guidelines to follow.

<img src="https://camo.githubusercontent.com/37a904cb209cbc07b9b58b281dc1407e91210983491bec069b61b453d1de8ec4/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f6d7264626f75726b652f6c6561726e2d68662d696d616765732f7265736f6c76652f6d61696e2f6c6561726e2d68662d746578742d636c617373696669636174696f6e2f30312d68756767696e672d666163652d776f726b666c6f772e706e67"
width="800">


## Importing necessary libraries

- **Library Imports & Setup**

  - Ensure correct setup by following the [setup guide](https://www.learnhuggingface.com/extras/setup) for local environments.
  - Google Colab users have most libraries pre-installed but need additional installations.
  - Enable GPU in Google Colab via `Runtime` ➡️ `Change runtime type` ➡️ `Hardware accelerator` ➡️ `GPU`.

- **Required Libraries from Hugging Face Ecosystem**

  - [`transformers`](https://huggingface.co/docs/transformers/en/installation) – Pre-installed on Colab; install with `pip install transformers`.
  - [`datasets`](https://huggingface.co/docs/datasets/installation) – Handles dataset access and manipulation; install with `pip install datasets`.
  - [`evaluate`](https://huggingface.co/docs/evaluate/installation) – Provides performance evaluation metrics; install with `pip install evaluate`.
  - [`accelerate`](https://huggingface.co/docs/accelerate/basic_tutorials/install) – Optimizes ML model training; install with `pip install accelerate`.
  - [`gradio`](https://www.gradio.app/guides/quickstart#installation) – Builds interactive ML model demos; install with `pip install gradio`.

- **Checking Installed Versions**
  - Use `package_name.__version__` to verify installed package versions.

## Getting a dataset

- **Dataset Importance in Machine Learning**

  - The dataset choice directly influences the model type and output quality.
  - High-quality datasets lead to better model performance, while poor datasets degrade model quality.

- **Text Classification Dataset Structure**

  - Typically consists of text samples (e.g., sentences, paragraphs) and corresponding labels.
  - The example dataset contains synthetic image captions labeled as "food" or "not food".

- **Dataset Source**

  - Available on Hugging Face: [`mrdbourke/learn_hf_food_not_food_image_captions`](https://huggingface.co/datasets/mrdbourke/learn_hf_food_not_food_image_captions).
  - Designed for practicing text classification tasks.

<img src="https://camo.githubusercontent.com/459a52d9b0e4ed339be867ee97adf4aead5b88fed338fa3d2af661b31874ace9/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f6d7264626f75726b652f6c6561726e2d68662d696d616765732f7265736f6c76652f6d61696e2f6c6561726e2d68662d746578742d636c617373696669636174696f6e2f6c6561726e2d68662d666f6f642d6e6f742d666f6f642d746578742d636c617373696669636174696f6e2d646174617365742e706e67"
width="800">

## Food Not Food Image Caption Dataset Creation

- **Food Not Food Image Caption Dataset Creation**

  - Dataset creation process documented in this [Google Colab notebook](https://colab.research.google.com/drive/14xr3KN_HINY5LjV0s2E-4i7v0o_XI3U8?usp=sharing).

- **Synthetic Data Generation**

  - A **Large Language Model (LLM)** was used to generate food and non-food image captions.
  - This technique is useful for **bootstrapping** dataset creation when real data is limited.
  - Recommended workflow: prioritize **real data** and supplement with **synthetic data** when necessary.

- **Model Evaluation Best Practices**
  - Always **evaluate and test models on real-world data** rather than relying solely on synthetic data.

### Where can you get more datasets?

- **Sources for Text-Based Datasets**

  - [Hugging Face Hub](https://huggingface.co/datasets) – A vast collection of datasets for various NLP tasks.
  - [Hugging Face Text Classification Datasets](https://huggingface.co/datasets?task_categories=task_categories:text-classification&sort=trending) – Specific datasets for text classification.
  - [Kaggle Datasets](https://www.kaggle.com/datasets) – A popular platform for diverse machine learning datasets.

- **Synthetic Dataset Creation with LLMs**
  - **Large Language Models (LLMs)** can generate synthetic data for text classification problems.
  - Enables custom dataset creation when real-world data is limited.
 
<img src="https://camo.githubusercontent.com/f2e6969ea6048f5232caf455f9e34b3025ac8ce0f2a7cb42465ca30cc5462e6f/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f6d7264626f75726b652f6c6561726e2d68662d696d616765732f7265736f6c76652f6d61696e2f6c6561726e2d68662d746578742d636c617373696669636174696f6e2f30332d706c616365732d746f2d6765742d746578742d64617461736574732e706e67"
width="800">

## Setting up a model for training

- **Steps for Model Training Setup**

  1. **Preprocess Data** – Prepare and clean dataset.
  2. **Define Model** – Use [`transformers.AutoModelForSequenceClassification`](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForSequenceClassification).
  3. **Set Training Arguments** – Configure hyperparameters with [`transformers.TrainingArguments`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).
  4. **Initialize Trainer** – Pass `TrainingArguments` and dataset to [`transformers.Trainer`](https://huggingface.co/docs/transformers/en/main_classes/trainer).
  5. **Train the Model** – Call [`Trainer.train()`](https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/trainer#transformers.Trainer.train).
  6. **Save the Model** – Store locally or on Hugging Face Hub.
  7. **Evaluate Performance** – Make predictions and analyze test data results.
  8. **Deploy Model** – Create a shareable demo.

- **Using Pretrained Models for Transfer Learning**

  - Load models with [`from_pretrained()`](https://huggingface.co/docs/transformers/v4.42.0/en/model_doc/auto#transformers.AutoConfig.from_pretrained).
  - **Pretrained Model:** [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased), trained on [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) and [English Wikipedia](https://huggingface.co/datasets/legacy-datasets/wikipedia).
  - **Transfer Learning Benefits:**
    1. Achieves good results with limited data.
    2. Can be adapted across various domains (e.g., vision, audio, NLP).
  - **Key Question:** "Does a pretrained model exist for my task, and can I fine-tune it?"
 
  <img src="https://camo.githubusercontent.com/02df5285232360f5702f9eb3b82a19de34882bd0c3634dc71ab359e1dbd4e6e4/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f6d7264626f75726b652f6c6561726e2d68662d696d616765732f7265736f6c76652f6d61696e2f6c6561726e2d68662d746578742d636c617373696669636174696f6e2f30392d7472616e736665722d6c6561726e696e672d6578616d706c652e706e67"
width="800">

- **Model Customization & Configuration**

  - Use `AutoModelForSequenceClassification` for text classification.
  - Customize model architecture with `pretrained_model_name_or_path` and `num_labels`.
  - Adjust classification head with [`transformers.PretrainedConfig`](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/configuration#transformers.PretrainedConfig) to set `id2label` and `label2id`.

- **Further Learning**
  - Example of **Transfer Learning in PyTorch**: [PyTorch Transfer Learning Guide](https://www.learnpytorch.io/06_pytorch_transfer_learning/).

  <img src="https://camo.githubusercontent.com/de066ed72039f3a831693348da2a84307ad7642c1918632b9e837eae8df7be9a/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f6d7264626f75726b652f6c6561726e2d68662d696d616765732f7265736f6c76652f6d61696e2f6c6561726e2d68662d746578742d636c617373696669636174696f6e2f30372d6c6561726e696e672d616e642d7472616e736665722d6c6561726e696e672d706172616469676d732e706e67"
width="800">


<a id="acknowledgments"></a>
## ![Credits Badge](https://img.shields.io/badge/Credits-DanielBourke-blue?style=flat-square)

The content is based on Daniel's comprehensive deep learning course `Text Classification with Hugging Face Transformers` and reflects his expertise in making complex deep learning concepts accessible through practical, hands-on examples.

Visit [Daniel's GitHub profile](https://github.com/mrdbourke) for more resources on machine learning and deep learning.
