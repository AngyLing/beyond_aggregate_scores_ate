"""GENIA labels shared by training and LLM parsing; no import side effects."""

GENIA_CLASSES = [
    "DNA_N/A",
    "DNA_domain_or_region",
    "DNA_family_or_group",
    "DNA_molecule",
    "DNA_substructure",
    "RNA_N/A",
    "RNA_domain_or_region",
    "RNA_family_or_group",
    "RNA_molecule",
    "RNA_substructure",
    "amino_acid_monomer",
    "atom",
    "body_part",
    "carbohydrate",
    "cell_component",
    "cell_line",
    "cell_type",
    "inorganic",
    "lipid",
    "mono_cell",
    "multi_cell",
    "nucleotide",
    "other_artificial_source",
    "other_name",
    "other_organic_compound",
    "peptide",
    "polynucleotide",
    "protein_N/A",
    "protein_complex",
    "protein_domain_or_region",
    "protein_family_or_group",
    "protein_molecule",
    "protein_substructure",
    "protein_subunit",
    "tissue",
    "virus",
]


def genia_coarse6(label):
    """Collapse a fine GENIA class to the 6-class scheme — used for the coarse
    arm's prompt set and for collapsing gold when scoring that arm."""
    if not label:
        return label
    for prefix in ("protein", "DNA", "RNA"):
        if label.startswith(prefix):
            return prefix
    if label in ("cell_line", "cell_type"):
        return label
    return "other"
