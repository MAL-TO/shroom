# SHROOM trial set
This data corresponds to the trial data for the SHROOM task 6 at Semeval 2024 (Shared-task on Hallucinations and Observable Overgeneration Mistakes).

## What is SHROOM?
The task consists in a binary classification, where participants are asked to determine whether a given production from an NLP model constitutes a hallucination

Participants will be ranked along two metrics: (i) accuracy and (ii) how well their probability correlates with the empirical probabilities observed in our annotators.

## File format
The file is formatted as a JSON list. Each element in this list corresponds to a datapoint.

Each datapoint corresponds to a different model production, and contains the following information:
- a task (`task`), indicating what objective the model was optimized for;
- a source (`src`), the input passed to the models for generation;
- a target (`tgt`), the intended reference "gold" text that the model ought to generate;
- a hypothesis (`hyp`), the actual model production;
- a set of per annotator labels (`labels`), indicating whether each individual annotator thought this datapoint constituted a hallucination or not;
- a majority-based gold-label (`label`), based on the previous per-annotator labels;
- a probability assigned to this datapoint being a hallucination (`p(Hallucination)`), corresponding to the proportion of annotators who considered this specific datapoint to be a hallucination.

We also include an indicator of whether target or source should serve as a semantic reference (`ref`): in some NLP tasks, such as Definition Modeling, the source may not contain the information necessary to establish whether the model production is factual wrong whereas in other cases, such as with Text Simplification, the same holds for the target. The `ref` key therefore indicate whether target, source or both of these fields contain the semantic information necessary to establish whether a datapoint is a hallucination.

Lastly, some datapoints also identify the model, as a huggingface identifier (`model`).

#### Example: interpreting a Definition Modeling (DM) datapoint

he definition modeling task was introduced in [Noraset et al (2017)](https://dl.acm.org/doi/10.5555/3298023.3298042). In this task, models are trained to generate a definition for a given example in context.

Here, we are specifically using the scheme of [Bevilacqua et al (2020)](https://aclanthology.org/2020.emnlp-main.585/). The source (`"src"`) corresponds to the context; the word to define is indicated using two special tokens `<define>` ... `</define>`.The target (`"tgt"`) is the intended definition for this context (as found in wiktionary); the hypothesis (`"hyp"`) is the actual model production.

To take a concrete example, the following datapoint in the trial set:

```json
    {
        "hyp": "(uncountable) The study of trees.",
        "ref": "tgt",
        "src": "It is now generally supposed that the forbidden fruit was a kind of citrus , but certain facts connected with <define> arborolatry </define> seem to me to disprove this opinion .",
        "tgt": "The worship of trees.",
        "model": "",
        "task": "DM",
        "labels": [
            "Hallucination",
            "Hallucination",
            "Hallucination"
        ],
        "label": "Hallucination",
        "p(Hallucination)": 1.0
    }
```

This corresponds to defining the word "arborolatry" (delinated by the `<define>` and `</define>` control tokens) in the following context (corresponding to the `src` key) : 
 + _It is now generally supposed that the forbidden fruit was a kind of citrus , but certain facts connected with arborolatry seem to me to disprove this opinion._

The model produced the following hypothesis ('hyp' key):
 + `(uncountable) The study of trees.`
 
whereas the gold definition from wiktionary ('tgt' key) is as follows:
 + _The worship of trees._

Annotators then marked whether this production is considered a hallucination or not. To do so, we asked them to study whether the hypothesis (`hyp` key) contains information that is not supported by the reference. Here, the `ref` key indicates that this reference corresponds to the target (given by its value, `"tgt"`). All three annotators considered the production to be a hallucination (cf. the `labels` key).

#### Example: interpreting a Paraphrase Generation (PG) datapoint

The same structure holds for the paraphrase generation (PG) task. For an example, consider the following trial datapoint:

```json
    {
        "hyp": "When did you see him?",
        "ref": "either",
        "src": "When\u2019d you last see him?",
        "tgt": "When was the last time you saw him?",
        "model": "tuner007/pegasus_paraphrase",
        "task": "PG",
        "labels": [
            "Not Hallucination",
            "Not Hallucination",
            "Not Hallucination"
        ],
        "label": "Not Hallucination",
        "p(Hallucination)": 0.0
    }
```

Using the following input (`src` key):
 + _When’d you last see him?_

the  model production (listed under the `hyp` key) was as follows:
 + `When did you see him?`

whereas the intended gold target (`tgt` key) was:
 + _When was the last time you saw him?_
 
All three annotators did not consider this production as hallucinatory (cf. the `labels` key). To do so, they were instructed to look whether all information stated in the hypothesis was supported by either/both the source and the target (as explicited with the `"either"` value of the `ref` key).

For PG datapoints, we also indicate the huggingface model that was used to generate the hypothesis, see the `model` key. 

#### Example: interpreting a Machine Translation (MT) datapoint

The structure of MT datapoints is consistent with PG and DM. For instance:

```json
    {
        "hyp": "I have nothing to do with it.",
        "ref": "either",
        "src": "J'en ai rien \u00e0 secouer.",
        "tgt": "I don't give a shit about it.",
        "model": "",
        "task": "MT",
        "labels": [
            "Hallucination",
            "Not Hallucination",
            "Hallucination"
        ],
        "label": "Hallucination",
        "p(Hallucination)": 0.6666666666666666
    }
```

In the above datapoint, the model was tasked with translating the source (`src`) "_J'en ai rien à secouer._"; the expected target gold translation (`tgt`) was "_I don't give a shit about it._"

Instead, the model produced the following (`hyp`):
+ `I have nothing to do with it.`

Two out of three annotators considered this production a hallucination (`labels` key), based either/both the source and the target (as explicited with the `"either"` value of the `ref` key). The majority label (`label` key) is therefore `"Hallucination"`. 

## How will this trial dataset differ from the train, validation and evaluation sets?
The trial set covers datapoints from definition modeling (DM), machine translation (MT) and paraphrase generation (PG). All other sets should also include text simplification (TS) datapoints.

Furthermore:
- The train set will not contain manual annotations.
- The validation and evaluation sets will involve five annotators per datapoint.


