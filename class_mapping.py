NAME2SUBTYPELABELS_MAP = {
    "BENIGN": 1, # unable to cross reference with the roi dataset from BRACS
    "Benign sure": 1,
    "Benign-sure": 1,
    "Pathological-benign (Benign-sure)": 1, # unable to cross reference with the roi dataset from BRACS
    "Pathologica benign": 2,
    "Pathological-benign": 2,
    "UDH": 3,
    "UDH-sure": 3,
    "FEA": 4,
    "FEA-sure": 4,
    "ADH": 5,
    "ADH-sure": 5,
    "DCIS": 6,
    "DCIS-sure": 6,
    "MALIGNANT": 7,
    "Malignant": 7,
    "Malignant-sure": 7
}

LABELS2SUBTYPE_MAP = {
    1: "Type_N",
    2: "Type_PB",
    3: "Type_UDH",
    4: "Type_FEA",
    5: "Type_ADH",
    6: "Type_DCIS",
    7: "Type_IC"
}

NAME2TYPELABELS_MAP = {
    "BENIGN": 1, # unable to cross reference with the roi dataset from BRACS
    "Benign sure": 1,
    "Benign-sure": 1,
    "Pathological-benign (Benign-sure)": 1, # unable to cross reference with the roi dataset from BRACS
    "Pathologica benign": 1,
    "Pathological-benign": 1,
    "UDH": 1,
    "UDH-sure": 1,
    "FEA": 2,
    "FEA-sure": 2,
    "ADH": 2,
    "ADH-sure": 2,
    "DCIS": 3,
    "DCIS-sure": 3,
    "MALIGNANT": 3,
    "Malignant": 3,
    "Malignant-sure": 3
}

LABELS2TYPE_MAP = {
    1: "Non-cancerous",
    2: "Pre-cancerous",
    3: "Cancerous"
}