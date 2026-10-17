# Rumor Detection · Research Prototype

**Django · Python · LSTM · Text preprocessing**

A historical academic project exploring rumor detection in Twitter-style conversations with sequence models and a Django interface. The repository brings together tweet collection, preprocessing, model experiments and research documentation.

## Explore the project

| Location | Purpose |
| --- | --- |
| [RumorDetection](RumorDetection) | Django views and application routes |
| [Detection](Detection) | Django project configuration |
| [latest111.py](latest111.py) | Sequence-model experiment |
| [extract_ids.py](extract_ids.py) | Tweet collection helper |
| [templates](templates) | Web interface templates |
| [Documentation](Documentation) | Project reports and research material |

## Reproducibility and setup

This is a legacy research snapshot. A complete pinned dependency environment is not included, and the older Django, Keras, Gensim and Tweepy APIs need compatibility review before execution.

Review the imports, dataset assumptions and model artifact provenance before rebuilding an isolated environment. The included `venv/`, database and serialized model are historical artifacts rather than a portable installation method. Do not load serialized model files from an untrusted source.

## API configuration

The collection code reads these variables from the environment:

```text
TWITTER_ACCESS_TOKEN
TWITTER_ACCESS_TOKEN_SECRET
TWITTER_CONSUMER_KEY
TWITTER_CONSUMER_SECRET
```

Set your own credentials locally. Previously committed API credentials must be revoked or rotated; removing values from current files does not remove them from Git history. Access to contemporary provider APIs and endpoints requires separate validation.

## Research scope

The original project explores LSTM sequence modeling with pooling for rumor-related text. No current benchmark accuracy, deployed fact-checking service or independently reproduced result is claimed here.

The original README referenced [this publication page](http://www.testmagzine.biz/index.php/testmagzine/article/view/3951). Publication metadata and the linked page have not been independently revalidated during this update.

## Project status

Historical academic prototype retained for source and research inspection. Runtime dependencies, external data access and end-to-end behavior require modernization and testing before reuse.

By [Datta Dhebe](https://github.com/DattaDhebe).
