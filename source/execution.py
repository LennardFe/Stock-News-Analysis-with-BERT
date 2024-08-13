from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
from nbformat import read
import time, os, asyncio
from tqdm import tqdm

# prevent warning message on windows
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# prevent warning message regarding frozen modules
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# overwrite preprocess_cell method to add progress bar
class ProgressExecutePreprocessor(ExecutePreprocessor):
    """
    Custom ExecutePreprocessor with progress bar support for code cells.

    Args (from parent class):
    - timeout (int, optional): Maximum execution time for each cell. Default is -1 (no timeout).
    - *args, **kwargs: Additional arguments passed to the parent class.

    Attributes:
    - progress_bar (tqdm): Progress bar instance.

    Methods:
    - preprocess_cell(cell, resources, index): Override of the parent method to execute code cells and update the progress bar.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_bar = None

    def preprocess_cell(self, cell, resources, index):
        """
        Override of the parent method to execute code cells and update the progress bar.

        Args:
        - cell (NotebookCell): The cell to be processed.
        - resources (dict): Additional resources used in the preprocessing.
        - index (int): Index of the cell in the notebook.

        Returns:
        - tuple: Tuple containing the processed cell and resources.
        """
        if cell.cell_type == "code": # only execute code cell and update progress bar
            result = super().preprocess_cell(cell, resources, index)
            if self.progress_bar:
                self.progress_bar.update(1)
            return result
        return cell, resources

def execute_notebook(notebook_path, save_name):
    """
    Execute a Jupyter notebook and save the results as an HTML file.

    Args:
    - notebook_path (str): Path to the Jupyter notebook file.
    - save_name (str): Desired name for the saved HTML file.

    Returns:
    - None
    """
    start_time = time.time()  # start the timer

    # read the notebook
    with open(notebook_path) as f:
        nb = read(f, as_version=4)

    # calculate the total number of code cells
    total_cells = sum(1 for cell in nb.cells if cell.cell_type == "code")

    # create the progress bar
    with tqdm(total=total_cells, desc="Executing notebook", unit="cell") as pbar:
        # execute the notebook with our custom preprocessor
        executor = ProgressExecutePreprocessor(timeout=-1)
        executor.allow_errors = True
        executor.progress_bar = pbar
        executor.preprocess(nb, {"metadata": {"path": "."}})

    # export the notebook as HTML
    html_exporter = HTMLExporter()
    (body, _) = html_exporter.from_notebook_node(nb)

    # create the results folder if it doesn't exist
    output_directory = "results"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # set output file path in the folder
    base_filename = os.path.splitext(os.path.basename(notebook_path))[0]
    output_file_name = f"{base_filename}_{save_name}"
    output_path = os.path.join(output_directory, output_file_name)

    # save the executed notebook as HTML in results
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(body)

    end_time = time.time()  # stop the timer
    execution_time = end_time - start_time  # calculate execution time

    # print execution summary
    print(f"Notebook executed successfully in {execution_time:.2f} seconds! HTML file saved at: {output_path}")