# UniDepth Inference Usage

First, install the UniDepth git repository via:

```sh
pip install git+https://github.com/lpiccinelli-eth/UniDepth.git
```

Then run the `inference.py` file using:

```sh
python -m inference --input_dir=<dataset_path> --task=<Train, or Test> --output_dir=<output_path>
```

Replace `<dataset_path>` with the path to your dataset, `<Train, or Test>` with either `Train` or `Test`, and `<output_path>` with the desired output directory.

# Depth Anything Usage

In terminal, simply run the command with the correct file paths:

python run_depth_anything_v2.py 
  --encoder vitl 
  --img-path <path> --outdir <outdir> 
 
