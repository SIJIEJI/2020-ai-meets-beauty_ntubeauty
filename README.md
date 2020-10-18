# AI Meets Beauty in ACM Multimedia 2020 3rd Place Solution
3rd place solution of the Grand Challenge of AI Meets Beauty in ACM Multimedia 2020 by team NTU-Beauty with mAP@7 0.405302

Part of the codes and pretrained models are originated from https://github.com/gniknoil/Perfect500K-Beauty-and-Personal-Care-Products-Retrieval-Challenge

## Citation

```
@inproceedings{hou2020attention,
  title={Attention-driven Unsupervised Image Retrieval for Beauty Products with Visual and Textual Clues},
  author={Hou, Jingwen and Ji, Sijie and Wang, Annan},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={4718--4722},
  year={2020}
}
```

## Guidance
* Prepare images: https://challenge2020.perfectcorp.com/
* Environment set up: `conda env create -f environment.yml` (have some redundant packages)
* Prepare pretrained models: download from https://github.com/gniknoil/Perfect500K-Beauty-and-Personal-Care-Products-Retrieval-Challenge and put `densenet201.t7` and `seresnet152.t7` under `./pretrained`
* Feature extraction: change the path to images in `features.py`, then `python features.py`. Results will be saved under `./feature`.
* Search: `python search_post_opt.py --test_image_path=<path_to_test_images> --result_key=<name_of_output_csv>`
