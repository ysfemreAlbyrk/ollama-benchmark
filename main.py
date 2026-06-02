import sys
import os

# Add the package parent directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ollamabenchmark.benchmark import main

if __name__ == "__main__":
    main()
