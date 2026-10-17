#!/usr/bin/env python
"""
Entrypoint for the academic profile collection pipeline.
Allows running the pipeline directly from the root workspace directory.
"""
import sys

if __name__ == "__main__":
    try:
        from academic_pipeline.cli import main
        sys.exit(main())
    except ImportError as e:
        print(f"Error: {e}")
        print("Please ensure dependencies are installed. You can install them with:")
        print("  pip install -r requirements.txt")
        print("Or by installing the package in editable mode:")
        print("  pip install -e .")
        sys.exit(1)
