import argparse
import re

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--input_path",
    type=str
)
argparser.add_argument(
    "--input_folder",
    type=str
)
argparser.add_argument(
    "--output_folder",
    help="Directory containing output files",
    type=str
)