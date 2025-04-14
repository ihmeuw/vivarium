from pathlib import Path

from vivarium import InteractiveContext
from vivarium.examples.disease_model.disease import (
    DiseaseModel,
    DiseaseState,
    DiseaseTransition,
    SISDiseaseModel,
)
from vivarium.examples.disease_model.intervention import TreatmentIntervention
from vivarium.examples.disease_model.mortality import Mortality
from vivarium.examples.disease_model.observer import DeathsObserver, YllsObserver
from vivarium.examples.disease_model.population import BasePopulation
from vivarium.examples.disease_model.risk import Risk, RiskEffect


def get_model_specification_path() -> str:
    p = Path(__file__).parent / "disease_model.yaml"
    return str(p)


def get_disease_model_simulation() -> InteractiveContext:
    p = get_model_specification_path()
    return InteractiveContext(p)
