import argparse
import Grid.GridGeneration as GridGenerator

parser = argparse.ArgumentParser(description="Grid Generation.")
parser.add_argument(
    "--content",
    type=str,
    default="./Data/sample/Content/Image",
    help="path to the folder of content images",
)
parser.add_argument(
    "--max_dim", type=int, default=900, help="Maximum dimension of the grid resolution"
)
parser.add_argument(
    "--max_ite", type=int, default=5, help="Maximum number of iterations"
)

if __name__ == "__main__":
    args = parser.parse_args()
    GridGenerator.StructureGrid(args)
