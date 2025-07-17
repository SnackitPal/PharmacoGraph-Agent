from tdc.utils import retrieve_dataset_names
for i in ['ADME', 'Tox', 'HTS', 'DrugRes', 'DDI', 'MolGen', 'LeadOpt', 'MultiPred', 'Genomics', 'Antibiotics']:
    print(i, retrieve_dataset_names(i))
