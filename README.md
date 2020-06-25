# AI Meets Beauty in ACM Multimedia 2020 3rd Place Solution
3rd place solution of the Grand Challenge of AI Meets Beauty in ACM Multimedia 2020 by team NTU-Beauty with mAP@7 0.405302

Part of the codes and pretrained models are originated from https://github.com/gniknoil/Perfect500K-Beauty-and-Personal-Care-Products-Retrieval-Challenge

## Citation
```
{TBD}
```

## Guidance
* Prepare images: https://challenge2020.perfectcorp.com/
* Environment set up: `conda env create -f environment.yml` (have some redundant packages)
* Prepare pretrained models: download from https://github.com/gniknoil/Perfect500K-Beauty-and-Personal-Care-Products-Retrieval-Challenge and put `densenet201.t7` and `seresnet152.t7` under `./pretrained`
* Feature extraction: change the path to images in `features.py`, then `python features.py`. Results will be saved under `./feature`.
* Search: `python search_post_opt.py --test_image_path=<path_to_test_images> --result_key=<name_of_output_csv>`