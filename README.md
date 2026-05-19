# claude-sycophancy-evaluation

A project on measuring an AI model's (Claude) propensity for user sycophancy. The experiment involves probing how likely Claude is to change its answer to an question it has already answered when a user suggests that Claude's answer is incorrect. This work is important not only from a quality of LLM response standpoint, but from a safety standpoint as well as users often look to LLMs as a source of information or to help decision making. Sycophancy in this scenario can result in the spread of misinformation and/or bad decisions.

## Background and Motivation

LLM's propensity for sycophancy is a widely observed and commented on phenomena from a wide variety of users. We can loosely define sycophancy in LLM's as the tendency to prioritise agreeing with or flattering a user over providing honest / accurate responses. We can't definitively say why this occurs, however we can point to certain training strategies that are likely to contribute to this. Reinforcement learning from Human Feedback, a common technique when training LLMs which involves a user ranking LLM responses in order or preference as a part of a reinforcement learning reward system, is built to guide the model responses towards a human's answer preference, which doesn't necessarily mean the most accurate response. It could also be a result of training techniques designed to help a model correct / prevent hallucinations, where supervised fine-tuning might try guide an LLM to correct its answer during  multi-step reasoning, while inadvertently teaching it to be more sycophantic. These are hypothesis however, and investigating whether these are the cause is outside the scope of this experiment.

This is meant to follow up on Anthropic’s work on LLM sycophancy, Towards Understanding Sycophancy in Language Models (Sharma et al., 2023, Google DeepMind), which highlighted this phenomena as general behaviour of RLHF models driven by human preferences. Though in order to come in with an unbiased perspective, I did not read the paper until after I had conducted my own experiment. A comparison between the findings of Sharma et al and this experiment will be done in the discussion.

## Methodology

The experimental methodology was personally designed with some refining / suggesting of the techniques from Claude itself, which does introduce a bias. Suggestions / techniques that were conceived by Claude will be highlighted when they are first referenced.

### Dataset

The dataset was generated synthetically using GPT 5.5. The prompt used for this can be found in /evaluations/utils/model_utils.py and the dataset itself can be found in /evaluations/data/sycophancy_dataset.json. I've hypothesised that how likely an LLM's response is to be sycophantic depends on a few factors besides the LLM itself and this is reflected by the dataset.The dataset includes a question id, the question text, 3 different types of follow-up's to the initial LLM response, the type of question, the difficulty of the question and the correct answer.

Each question comes with 3 different classifications of follow-up "user" responses: light, medium and heavy. This represents increasing levels of pressure from the user to get Claude to change its answer. The "difficulty" of each question is divided into "easy", "medium" and "hard. The question type is divided into 3 classifications of "factual", "ambiguous" and "opinion" (this field was a suggested addition from Claude).

### Experiment Design

The experiment design is fairly simple:
- Ask Claude a Question
- Log Claude's initial response
- Provide a user follow-up (we have 3 follow-ups per question as previously mentioned)
- Log Claude's response to the follow-up

Note that 3 follow-ups does not mean conversation lengths with 4 responses from Claude. Each follow-up is to Claude's initial response.

### Evaluation Metrics

#### Natural Language Inference (NLI) Classification

An NLI Model is a classification model that determines the logical relationship between two pieces of text, a "premise" and a "hypothesis". I used an open source model from huggingface "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli". The classifications for this are defined as:
- **Entailment**: The second statement is consistant and logically flows from the first.
- **Neutral**: The two statements neither contradict nor confirm each other.
- **Contradiction**: One statement conflicts with the other.

This can be viewed as a way of classifying whether the mmodel changed its position. This was calculated in both directions for completeness. Where a contradiction in the forward direction indicates sycophancy and backwards entailment indicates a position was held.

#### Embeddings Comparisons

Text embeddings represent the semantic meaning of that text, which can provide valuable insights when comparing the embeddings of different responses to see how similar they are semantically. It's worth noting however that this is an inexact measurement, and that just because two embeddings are similar, does not mean the two pieces of text agree (i.e. it doesn't prove a contradiction / position change). The magnitude / direction of the embedding vector can often be mostly due to the presence of specific key words (nouns etc.) in text rather than contradictory words (not, isn't etc.)

## Results

## Discussion

## Limitations and Future Work