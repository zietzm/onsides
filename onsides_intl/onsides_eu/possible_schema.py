from sqlmodel import Field, Relationship, SQLModel


class DrugProduct(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ema_product_number: str
    product_name: str
    rxnorm_code: str | None
    rxnorm_name: str | None
    num_ingredients: int

    ingredients: list["Ingredient"] = Relationship(back_populates="drugs")
    adverse_effects: list["AdverseEffect"] = Relationship(back_populates="drugs")


class Ingredient(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ingredient_name: str
    rxnorm_code: str | None
    rxnorm_name: str | None

    drugs: list[DrugProduct] = Relationship(back_populates="ingredients")


class DrugAdeLink(SQLModel, table=True):
    drug_id: int | None = Field(default=None, foreign_key="drug.id", primary_key=True)
    ade_id: int | None = Field(default=None, foreign_key="ade.id", primary_key=True)
    source: str
    section: str


class AdverseEffect(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    effect_name: str
    meddra_code: str | None
    meddra_name: str | None

    drugs: list[DrugProduct] = Relationship(
        back_populates="adverse_effects", link_model=DrugAdeLink
    )


###############################################################################
# Inputs
###############################################################################
class SourceIngredient(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    source_id: str
    source_name: str

    labels: list["SourceDrugLabel"] = Relationship(back_populates="ingredients")


class SourceDrugLabel(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    source_id: str
    source_name: str
    label_text: str

    ingredients: list[SourceIngredient] = Relationship(back_populates="labels")


###############################################################################
# Outputs
###############################################################################
class OutputIngredient(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    source_id: str
    source_name: str
    rxnorm_code: str | None
    rxnorm_name: str | None

    labels: list["OutputDrugLabel"] = Relationship(back_populates="ingredients")


class OutputAde(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    meddra_code: str | None
    meddra_name: str | None

    label: list["OutputDrugLabel"] = Relationship(back_populates="ade")


class OutputDrugLabel(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    source_id: str
    source_name: str
    rxnorm_code: str | None
    rxnorm_name: str | None

    ingredients: list[OutputIngredient] = Relationship(back_populates="labels")
    ade: list[OutputAde] = Relationship(back_populates="label")
