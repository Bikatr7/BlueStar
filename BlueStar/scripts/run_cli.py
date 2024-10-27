import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
bluestar_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, bluestar_dir)

from BlueStar.utils.cli_utils import main

if __name__ == "__main__":
    main()