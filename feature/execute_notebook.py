"""Execute the analysis notebook with the project's myenv kernel."""

from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
import jupyter_core.paths


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "feature" / "XGB_candidate_CN_feature_analysis.ipynb"
    # The managed Windows runtime rejects Jupyter's ACL hardening call.  The
    # execution still uses the explicit myenv kernel and only writes the
    # connection file to Jupyter's transient runtime directory.
    jupyter_core.paths.allow_insecure_writes = True
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=7200,
        kernel_name="myenv",
        resources={"metadata": {"path": str(project_root)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    print(f"Executed {notebook_path}")
    print(f"Kernel: {sys.executable}")


if __name__ == "__main__":
    main()
