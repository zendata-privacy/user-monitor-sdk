"""Vendored copy of ``pii-api/src/constants.py`` PII_TYPES.

The SDK has zero hard dependencies and cannot import the PII API package, but
``classification._assert_total_coverage()`` needs the authoritative list to
prove every entity type is mapped to a clearance class. Regenerate this file
whenever the PII API adds a type; the assertion fails at import until the
mapping catches up.
"""

PII_TYPES = [
    "ACCOUNT_NUMBER",
    "AGE",
    "DATE",
    "DATE_INTERVAL",
    "DOB",
    "DRIVER_LICENSE",
    "DURATION",
    "EMAIL_ADDRESS",
    "EVENT",
    "FILENAME",
    "GENDER_SEXUALITY",
    "HEALTHCARE_NUMBER",
    "IP_ADDRESS",
    "LANGUAGE",
    "LOCATION",
    "LOCATION_ADDRESS",
    "MARITAL_STATUS",
    "MONEY",
    "NAME",
    "NUMERICAL_PII",
    "OCCUPATION",
    "ORGANIZATION",
    "ORGANIZATION_MEDICAL_FACILITY",
    "ORIGIN",
    "PASSPORT_NUMBER",
    "PASSWORD",
    "PHONE_NUMBER",
    "PHYSICAL_ATTRIBUTE",
    "POLITICAL_AFFILIATION",
    "RELIGION",
    "SSN",
    "TIME",
    "URL",
    "USERNAME",
    "VEHICLE_ID",
    "ZODIAC_SIGN",
    "BLOOD_TYPE",
    "CONDITION",
    "DOSE",
    "DRUG",
    "INJURY",
    "MEDICAL_PROCESS",
    "STATISTICS",
    "BANK_ACCOUNT",
    "CREDIT_CARD",
    "CREDIT_CARD_EXPIRATION",
    "CVV",
]
