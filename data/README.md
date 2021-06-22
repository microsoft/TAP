## Data Organization
We recommend using the following AzCopy command to download.
AzCopy executable tools can be downloaded [here](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy).
Move ``GoogleCC`` folder under ``data`` to match the default paths.

[TextVQA/Caps/STVQA Data (~62G)](https://tapvqacaption.blob.core.windows.net/data/data).

[OCR-CC Data (Huge, ~1.3T)](https://tapvqacaption.blob.core.windows.net/data/GoogleCC).

[Model checkpoints (~17G)](https://tapvqacaption.blob.core.windows.net/data/save). 

A subset of OCR-CC with around 400K samples is availble in imdb ``data/imdb/cc/imdb_train_ocr_subset.npy``. The subset is faster to train with a small drop in performance, compared with the full set ``data/imdb/cc/imdb_train_ocr.npy``.

```
path/to/azcopy copy <folder-link> <target-address> --resursive"

# for example, downloading TextVQA/Caps/STVQA Data
path/to/azcopy copy https://tapvqacaption.blob.core.windows.net/data/data <local_path>/data --recursive

# for example, downloading OCR-CC Data
path/to/azcopy copy https://tapvqacaption.blob.core.windows.net/data/GoogleCC <local_path>/data/GoogleCC --recursive

# for example, downloading model checkpoints
path/to/azcopy copy https://tapvqacaption.blob.core.windows.net/data/save <local_path>/save --recursive
```
