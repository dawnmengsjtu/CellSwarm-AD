# -*- coding: utf-8 -*-
"""Public AD dataset index."""
from .schema import DatasetInfo

AD_DATASETS = [
    DatasetInfo(
        name="ADNI",
        url="https://adni.loni.usc.edu/",
        description="Alzheimer's Disease Neuroimaging Initiative - longitudinal MRI, PET, CSF, genetics",
        data_types=["MRI", "PET", "CSF biomarkers", "genetics", "cognitive scores"],
        species="human",
        access="application",
        citation="Mueller et al. (2005) Neuroimaging Clin N Am"
    ),
    DatasetInfo(
        name="AMP-AD",
        url="https://adknowledgeportal.synapse.org/",
        description="Accelerating Medicines Partnership for AD - multi-omics data",
        data_types=["RNA-seq", "proteomics", "metabolomics", "genetics"],
        species="human",
        access="public",
        citation="Greenwood et al. (2020) Nat Med"
    ),
    DatasetInfo(
        name="Allen Brain Atlas",
        url="https://portal.brain-map.org/",
        description="Gene expression atlas of the human and mouse brain",
        data_types=["ISH", "RNA-seq", "scRNA-seq", "spatial transcriptomics"],
        species="human/mouse",
        access="public",
        citation="Hawrylycz et al. (2012) Nature"
    ),
    DatasetInfo(
        name="AlzData",
        url="http://www.alzdata.org/",
        description="Multi-omics database for Alzheimer's disease",
        data_types=["GWAS", "eQTL", "gene expression", "methylation"],
        species="human",
        access="public",
        citation="Xu et al. (2018) Alzheimers Dement"
    ),
]
