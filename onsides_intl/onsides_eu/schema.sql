CREATE TABLE ingredient (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rxnorm_code TEXT,
    rxnorm_name TEXT,
);

CREATE TABLE adverse_effect (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meddra_code TEXT,
    meddra_name TEXT,
);

CREATE TABLE drug_product (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rxnorm_code TEXT,
    rxnorm_name TEXT,
    is_active BOOLEAN,
);

CREATE TABLE product_ingredients (
    drug_id INTEGER,
    ingredient_id INTEGER,
    FOREIGN KEY (drug_id) REFERENCES drug_product(id),
    FOREIGN KEY (ingredient_id) REFERENCES ingredient(id),
);

CREATE TABLE product_adverse_effects (
    drug_id INTEGER,
    effect_id INTEGER,
    score REAL,
    FOREIGN KEY (drug_id) REFERENCES drug_product(id),
    FOREIGN KEY (effect_id) REFERENCES adverse_effect(id),
);
