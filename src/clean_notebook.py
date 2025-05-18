import nbformat
from nbformat.v4 import new_notebook, new_code_cell

def clean_notebook(input_path, output_path):
    with open(input_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    new_nb = new_notebook()
    for cell in nb.cells:
        if cell.cell_type == 'code':
            new_nb.cells.append(new_code_cell(cell.source))
        else:
            new_nb.cells.append(cell)
    
    with open(output_path, 'w') as f:
        nbformat.write(new_nb, f)