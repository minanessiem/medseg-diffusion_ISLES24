#!/usr/bin/env python3
"""
Backwards compatibility wrapper for start_training.py.

This file maintains backwards compatibility for scripts and documentation
that reference main.py. All functionality has been moved to start_training.py.

Usage:
    python main.py --config-name <config>
    
    (equivalent to: python start_training.py --config-name <config>)

For new code, prefer using start_training.py directly.
For resuming interrupted runs, use resume_training.py.
"""

# Import and run the main function from start_training
from start_training import main

if __name__ == '__main__':
    main()
