# ECE 285 - MLIP - RNNs for Visual Question Answering
Pytorch implementation for Visual Question Answering for ECE285

## Demo

The demo utilizes the `demo.ipynb` file. The user must download the following files

[Weights](https://drive.google.com/open?id=1D82mDfuVhqLyNusSmZDzzfSUTTgf6u7d) - 
Place the weights in a file under `/models/demo.ckpt`

[Serialized data](https://drive.google.com/open?id=1JYtenpvkr5zUURVMyUoi0UV46ynA7mJo) - 
Place the file under `pickle/datapoint`

## Code Organization

- `/models` - Checkpoint directory
- `/pickle` - Serialized data directory
- `/utilities` - utilities directory
- `/logs` - Log directory
- `demo.ipynb` - Demo notebook
- `vqa.py` - Model Class
- `nntools.py` - Model Class helper
- `training.py` - Model Trainer
- `data_loader.py` - Data loader
- `text_helper.py` - Text helper

## Dataset Downloading

We used the infrastructure provided by GitHub user tbmoon's PyTorch implementation. [[repo]](https://github.com/tbmoon/basic_vqa). Here are the necessary commands:

```bash
$ cd utilities
$ chmod +x download_and_unzip_datasets.sh
$ ./download_and_unzip_datasets.sh
$ python resize_images.py --input_dir='../datasets/Images' --output_dir='../datasets/Resized_Images'  
$ python make_vacabs_for_questions_answers.py --input_dir='../datasets'
$ python build_vqa_inputs.py --input_dir='../datasets' --output_dir='../datasets'
```

## Model training.

```bash
$ python training.py
```



