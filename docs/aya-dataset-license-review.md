# Aya Dataset license review for Ryn SFT

Reviewed 2026-11-15. This review is pinned to `CohereLabs/aya_dataset` revision `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`; no dataset rows were downloaded.

## Decision

**Use for SFT, based on the publisher's license declaration.** At this revision, the dataset card declares `apache-2.0`, says the dataset may be used for any purpose including commercial use, and explicitly includes training and fine-tuning. The card states no non-commercial restriction, share-alike condition, or separate dataset use policy. This supports using the human-annotation data in Ryn SFT.

This is a dataset-level license conclusion, not a row-by-row rights clearance or a conclusion about the license for Ryn's trained weights.

## Terms to carry forward

The pinned repository root contains the card, data directories, `.gitattributes`, and header image, but no separate `LICENSE` or `NOTICE` file. The applicable license text is the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

- Apache 2.0 has no share-alike or non-commercial condition. Its redistribution terms require a copy of the license, prominent notices on modified files, retention of applicable source copyright, patent, trademark, and attribution notices, and preservation of notices from a `NOTICE` file if the source includes one. The pinned repository has no `NOTICE` file.
- The license grants limited patent rights, which terminate for the work if the recipient initiates patent litigation alleging that the work or a contribution infringes. It does not grant Cohere trademarks and supplies the work without warranty, including title and non-infringement warranties.
- The dataset card provides a BibTeX citation. Keep the dataset citation and pinned revision in Ryn's data/license manifest as provenance. Apache's notice duties apply when redistributing the dataset or a covered derivative; they are not a general requirement to display attribution in model responses. Whether a trained checkpoint is a covered derivative is not decided here.

## Ryn handling and caveats

- For SFT, use the default human-annotation configuration and its training split; exclude the held-out test split and the separate `demographics` configuration. The default records include a unique `user_id`; drop that field from training examples.
- The card describes original annotations and re-annotations of existing material. Cohere's paper likewise says Aya includes re-annotations of existing datasets. The public card does not provide record-level source licensing, and Apache 2.0 disclaims warranties of title and non-infringement. If Ryn's rights-cleared gate requires proof for each underlying source, hold dataset ingestion until that provenance is reviewed.
- Cohere's Aya model paper separately describes the **model** as Apache 2.0. That model license is not the basis for this dataset review; the pinned dataset card is.

## Sources

- [Hugging Face dataset card at pinned revision](https://huggingface.co/datasets/CohereLabs/aya_dataset/blob/f9ea04583f02a8f86404ff6c58bf75fe637df8a2/README.md)
- [Hugging Face repository tree at pinned revision](https://huggingface.co/datasets/CohereLabs/aya_dataset/tree/f9ea04583f02a8f86404ff6c58bf75fe637df8a2)
- [Apache License 2.0, official text](https://www.apache.org/licenses/LICENSE-2.0)
- [Cohere, “Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning”](https://cohere.com/research/papers/aya-dataset-paper-2024-02-13)
- [Cohere, “Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model”](https://cohere.com/research/aya/aya-model-paper.pdf)
