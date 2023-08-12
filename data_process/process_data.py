import numpy as np
import pandas as pd
import pyreadr
from pubchempy import get_compounds, Compound


gene_expression_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/lung_and_all_processed_data/CCLE/RNAseq.rds"
cell_line_drug_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug-CCLE.rds"


gene_expression = pyreadr.read_r(gene_expression_path)
cell_line_drug = pyreadr.read_r(cell_line_drug_path)



