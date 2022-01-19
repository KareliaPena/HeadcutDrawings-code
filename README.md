# HedcutDrawings: Rendering hedcut style portraits

We propose a new neural-based style transfer algorithm for the generation of hedcut drawings that uses side information to impose additional constraints on the direction of the dots.
 
## Setup
Prerequisites
- Windows
- Python 3.6.7

Create a conda environment using requirements.txt file.

 `conda create --name <env> --file <this file>`
 
## Generate the grid
First, create a folder for the data and include the desired content image in /path/to/data/folder/Content/Image. Then, run the grid generation script for the generation of the grid. The grid image will be stored in /path/to/data/folder/Content/Grid.

 ` python Grid_Generation.py --content /path/to/content/ --max_dim grid_resolution  --max_ite maximum_iterations`

## Generate the HeadcutDrawing


The segmentation of the content image is included in /path/to/data/folder/Content/Segmentation with the same name as the corresponding content image. Then run the script for the generation of the HedcutDrawing. The hedcut will be stored in /path/to/data/folder/output.

 ` python HedcutDrawing_Generation.py --data_folder /path/to/data/folder/ --max_ite maximum_iterations`
