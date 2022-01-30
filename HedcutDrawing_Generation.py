

import argparse
import Neural_Style.Style_main as StyleGenerator


parser = argparse.ArgumentParser(description='Grid Generation.')
parser.add_argument('--data_folder', type=str, default="./Data/sample/",
                    help="path to the folder of content images")
parser.add_argument('--max_ite', type=int, default=500,
                    help="Maximum number of iterations")

if __name__ == '__main__':
    args = parser.parse_args()
    StyleGenerator.HedcutDrawings(args)
     
    
    


