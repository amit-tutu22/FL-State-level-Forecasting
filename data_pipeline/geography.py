"""Source-aware names and conservative historical boundary handling.

Never join CPI and GST numeric codes: CPI 2024 codes are a different namespace.
Unknown or boundary-ambiguous names are retained but excluded from model joins.
"""

import re

CANONICAL = [
    "Andaman and Nicobar Islands",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Chhattisgarh",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu and Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Ladakh",
    "Lakshadweep",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Dadra and Nagar Haveli",
    "Daman and Diu",
    "Dadra and Nagar Haveli and Daman and Diu",
]


def key(name):
    """Normalize punctuation and whitespace without fuzzy geography matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower().replace("&", " and ")).strip()


NAMES = {key(name): name for name in CANONICAL}
NAMES.update(
    {
        key(alias): target
        for alias, target in {
            "Andaman and Nicobar Island": "Andaman and Nicobar Islands",
            "Andaman and Nicobar": "Andaman and Nicobar Islands",
            "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
            "NCT of Delhi": "Delhi",
            "Orissa": "Odisha",
            "Pondicherry": "Puducherry",
            "Uttaranchal": "Uttarakhand",
            "The Dadra And Nagar Haveli And Daman And Diu": (
                "Dadra and Nagar Haveli and Daman and Diu"
            ),
        }.items()
    }
)
NON_CLIENTS = {
    key(n)
    for n in [
        "All India",
        "Other Territory",
        "Other Territories",
        "CBIC",
        "OIDAR",
        "Center Jurisdiction",
        "Centre Jurisdiction",
    ]
}

# GST administrative codes, not CPI codes. Historical names remain separate.
GST_CODES = {
    1: "Jammu and Kashmir",
    2: "Himachal Pradesh",
    3: "Punjab",
    4: "Chandigarh",
    5: "Uttarakhand",
    6: "Haryana",
    7: "Delhi",
    8: "Rajasthan",
    9: "Uttar Pradesh",
    10: "Bihar",
    11: "Sikkim",
    12: "Arunachal Pradesh",
    13: "Nagaland",
    14: "Manipur",
    15: "Mizoram",
    16: "Tripura",
    17: "Meghalaya",
    18: "Assam",
    19: "West Bengal",
    20: "Jharkhand",
    21: "Odisha",
    22: "Chhattisgarh",
    23: "Madhya Pradesh",
    24: "Gujarat",
    25: "Daman and Diu",
    26: "Dadra and Nagar Haveli",
    27: "Maharashtra",
    29: "Karnataka",
    30: "Goa",
    31: "Lakshadweep",
    32: "Kerala",
    33: "Tamil Nadu",
    34: "Puducherry",
    35: "Andaman and Nicobar Islands",
    36: "Telangana",
    37: "Andhra Pradesh",
    38: "Ladakh",
}


def check_gst_code(code, name):
    """Validate the source GST code against its explicit name mapping."""
    if int(code) in {97, 99} and key(name) in NON_CLIENTS:
        return
    canonical = NAMES.get(key(name))
    allowed = {GST_CODES.get(int(code))}
    if int(code) == 26:
        allowed.add("Dadra and Nagar Haveli and Daman and Diu")
    if canonical is None or canonical not in allowed:
        raise ValueError(f"GST source code/name mismatch: {code}, {name}")


def resolve(name, month, source):
    """Return a stable ID, canonical name and historical boundary review status."""
    normalized = key(name)
    if normalized in NON_CLIENTS:
        return normalized.replace(" ", "_"), str(name).strip(), "non_client"
    if normalized not in NAMES:
        return "unmapped", str(name).strip(), "unmapped_name"
    canonical = NAMES[normalized]
    identifier = key(canonical).replace(" ", "_")
    status = "validated_name"
    if canonical in {"Jammu and Kashmir", "Ladakh"}:
        if month == "2019-10":
            status = "boundary_transition_month"
        elif month < "2019-10":
            if canonical == "Ladakh":
                status = "pre_formation_placeholder"
            else:
                identifier += "_undivided"
        elif source == "cpi_2012":
            # This source retains old coverage labels; no unsupported split inference.
            status = "boundary_definition_unverified"
    if canonical in {"Dadra and Nagar Haveli", "Daman and Diu"}:
        identifier += "_legacy"
        if month >= "2020-01":
            status = "boundary_definition_unverified"
    if canonical == "Dadra and Nagar Haveli and Daman and Diu" and month <= "2020-01":
        status = "boundary_transition_or_pre_formation"
    return identifier, canonical, status
