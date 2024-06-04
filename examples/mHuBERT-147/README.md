## mHuBERT-147: A Compact Multilingual HuBERT Model

This folder contain the templates for launching the mHuBERT-147 training.

## Data format

Different from the manifest from HuBERT/wav2vec2, for mHuBERT-147 the manifest is split in several files, to simplify pre-processing file management.

Two folders are needed: one named "train", another named "valid". Put your collection of manifest files, with assotiated labels, inside these folders (see example in data_example).

Each file contains the following:
* .tsv
    First line: path to the wav file
    Following lines: wav_file_name \t frames \t language_id \t dataset_name
* .km
    List of labels for the lines of the corresponding .tsv

There is no specific template for language id and dataset name, it is just important to be consistent across manifest files, as these tokens will be used as labels for computing the two-level language-dataset up-sampling for batching. More details about how the up-sampling is computed can be found at the source code at: fairseq/data/audio/hubert_dataset.py

## Preprocessing and faiss clustering

Check the pre-processing described in the mHuBERT-147 pre-processing repository:
https://github.com/utter-project/mHuBERT-147-scripts/

## Trained models and manifest files

Available at: PUT CITATION TO COLLECTION

## Citation

```
TO DO
```