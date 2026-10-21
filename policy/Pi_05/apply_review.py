#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from real_world.apply_review import main
if __name__ == '__main__':
    main()
