import os
import socket
import logging

# Prevent benign faiss AVX512 warning and silence third-party loader logs
if "FAISS_OPT_LEVEL" not in os.environ:
    os.environ["FAISS_OPT_LEVEL"] = "AVX2"
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

# Set a global default socket timeout to prevent synchronous network operations
# (like urllib robots.txt reads) from hanging the pipeline execution indefinitely.
socket.setdefaulttimeout(15.0)

from academic_pipeline.pipeline import AcademicPipeline
from academic_pipeline.report_generator import ReportGenerator

__all__ = ["AcademicPipeline", "ReportGenerator"]
