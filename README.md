# DiffHallu

DiffHallu is a model-agnostic implementation of differential testing for
hallucination-risk detection in long-form generation. It compares multiple
continuations produced by the same target model under the same prefix, groups
semantically equivalent continuations, and reports their average weighted
semantic entropy as a continuous risk score.

This repository provides the core method implementation. Supporting experimental
materials are available through the link below.

Chinese documentation: [README_zh-CN.md](README_zh-CN.md)

## Reproduction materials

[Download the DiffHallu reproduction materials from Google Drive](https://drive.google.com/drive/folders/12O1JY1aL4N-JBhYNvknP7U8ISY03HDMf?usp=sharing).

The materials include processed experiment data and fixed data splits, generated
candidates and truth labels, experiment and baseline scripts and configurations,
probe model artifacts, runtime environment and dependency version information,
experiment results, and reproduction tools. See the folder's README for its
structure and usage instructions.

## Method

For a prompt `x`, generated text `z`, and target entity `e`, DiffHallu:

1. locates the sentence containing `e` and constructs its left context;
2. builds DiffHallu's controlled-rewrite prompt and sends it through a
   user-provided API interface, moving `e` to the end of the sentence;
3. samples `M` continuations from a user-provided target-model backend;
4. obtains teacher-forced token log-probabilities for every continuation;
5. builds a bidirectional NLI equivalence matrix;
6. runs representative clustering under every cyclic scan order; and
7. averages weighted semantic entropy across the clustering results.

The output is a continuous hallucination-risk score. Higher scores indicate a
more dispersed semantic distribution among sampled continuations.

## Installation

```bash
pip install -e .
```

Python 3.10 or later is required. The only runtime dependency is `pysbd`, used
for English sentence segmentation.

## Model interfaces

DiffHallu does not select, configure, load, or deploy models. Users implement
three interfaces from `diffhallu.interfaces`:

- `RewriterBackend`: sends DiffHallu's built-in rewrite prompt to the user's
  own API and returns the API's raw text response. DiffHallu owns the prompt,
  JSON parsing, `[BLANK]` construction rules, and structural validation.
- `TargetModelBackend`: generates candidates and returns teacher-forced token
  log-probabilities for a requested completion.
- `SemanticBackend`: returns `entailment`, `neutral`, or `contradiction` for a
  directional pair of texts.

The prefix-rewrite stage therefore requires users to provide their own API
implementation, including its URL, authentication, retry policy, and selected
model. No credential or API configuration is stored in this repository.

The target-generation stage requires users to deploy the target model and
implement both generation and teacher-forced scoring. The deployed model must
expose token-level conditional log-probabilities for arbitrary supplied
continuations; a text-only generation endpoint is insufficient.

## Reference reproduction backends

The repository deliberately leaves all model implementations outside the
method package. To reproduce the model choices recorded in the paper:

- implement `RewriterBackend` with your own API. The paper's validity-analysis
  configuration records **Qwen3.5** as the auxiliary model used for target
  sentence adjustment. This is a reference reproduction choice, not a model
  required or configured by DiffHallu;
- implement `SemanticBackend` with **DeBERTa-v2-xxlarge-MNLI** and map its
  outputs to `entailment`, `neutral`, and `contradiction`.

API URLs, credentials, model paths, serving frameworks, and devices remain the
user's responsibility.

## Reference conversation templates

`diffhallu.templates` provides the reference conversation templates and
system prompts used by the four target models in the experiments. Supported
family keys are `qwen`, `llama-3`, `gemma`, and `mistral`.

- Llama 3 and Qwen use `You are a helpful assistant.`
- Gemma and Mistral templates do not add a separate system message.

```python
from diffhallu import COMMON_STOP_TOKENS, render_conversation_prompt

formatted_prompt = render_conversation_prompt(
    model_family="qwen",
    instruction=prompt,
    prefix=prefix,
)
```

The target-model backend may use these templates directly. If a deployed
checkpoint has an authoritative tokenizer chat template, users may instead
use that native template while keeping the same instruction and fixed prefix.
The exported `COMMON_STOP_TOKENS` tuple contains the reference stop markers.

```python
from diffhallu import DiffHallu


class MyRewriter:
    def complete(self, prompt, config):
        # Send DiffHallu's prompt to your own API and return the raw response.
        return call_my_api(
            prompt=prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )


class MyTargetModel:
    def generate(self, prompt, prefix, config):
        # Return exactly config.num_samples continuations.
        # Honor config.temperature, top_p, top_k, and max_new_tokens.
        # You may render prompt + prefix with diffhallu.templates.
        raise NotImplementedError

    def token_logprobs(self, prompt, prefix, completion):
        # Return one teacher-forced conditional log-probability per token.
        raise NotImplementedError


class MySemanticModel:
    def predict(self, premise, hypothesis):
        # Return entailment, neutral, or contradiction.
        raise NotImplementedError


detector = DiffHallu(
    rewriter=MyRewriter(),
    target_model=MyTargetModel(),
    semantic_model=MySemanticModel(),
)

result = detector.detect(
    prompt="Write a short biography.",
    generated_text="The report was written by Ada Lovelace.",
    target_entity="Ada Lovelace",
)

print(result.risk_score)
print(result.to_dict())
```

When an entity occurs more than once, pass `occurrence_index`, or pass the
exact `char_start` and `char_end` offsets.

## Default method parameters

| Parameter | Default |
| --- | ---: |
| rewrite API `max_tokens` | `768` |
| rewrite API `temperature` | `0.7` |
| `num_samples` | `10` |
| `max_new_tokens` | `50` |
| `do_sample` | `True` |
| `temperature` | `1.0` |
| `top_p` | `0.9` |
| `top_k` | `50` |

These are DiffHallu method defaults, not model deployment settings. The
rewrite and generation values can be overridden with `DiffHalluConfig`.

The following behaviors are fixed parts of the published method and are not
configuration options:

- lax bidirectional NLI equivalence: any contradiction rejects equivalence;
  two neutral predictions also reject it; otherwise the pair is equivalent;
- every generated continuation is reduced to its first complete sentence;
- representative clustering is run under all `M` cyclic scan orders, and the
  `M` weighted semantic entropies are averaged.

## Backend contract details

`TargetModelBackend.generate` must return exactly `num_samples` strings from
the same prompt and prefix. `TargetModelBackend.token_logprobs` must score only
the supplied completion under the same prompt and prefix. DiffHallu computes
the arithmetic mean of those token log-probabilities before constructing
semantic-cluster weights.

The target backend is responsible for chat templates, tokenization, stop-token
IDs, padding, devices, batching, and model-specific generation behavior.

## Dataset

The original method uses the entity-level annotations from
[obalcells/longfact-annotations](https://huggingface.co/datasets/obalcells/longfact-annotations).
For paper-aligned reproduction, use these four dataset configurations:

- `Qwen2.5-7B-Instruct`
- `Meta-Llama-3.1-8B-Instruct`
- `gemma-2-9b-it`
- `Mistral-Small-24B-Instruct-2501`

The dataset also currently exposes a `Llama-3.3-70B-Instruct` configuration;
it is not one of the four target models evaluated in the paper.

Map each dataset row to DiffHallu as follows:

```python
row = dataset_row
prompt = row["conversation"][0]["content"]       # user message
generated_text = row["conversation"][1]["content"]

for annotation in row["annotations"]:
    target_entity = annotation["span"]
    char_start = annotation["index"]
    char_end = char_start + len(target_entity)

    result = detector.detect(
        prompt=prompt,
        generated_text=generated_text,
        target_entity=target_entity,
        char_start=char_start,
        char_end=char_end,
    )
```

`annotation["index"]` is the character offset in the assistant message. The
dataset is not copied into this repository; download it from Hugging Face and
follow the terms published with it.

## Scope

DiffHallu requires multiple samples and conditional generation probabilities.
A text-only black-box model cannot reproduce the method unless its backend can
also provide equivalent teacher-forced scores. The default sentence segmenter
and semantic-equivalence contract target English factual content.
