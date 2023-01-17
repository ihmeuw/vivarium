from .disease import DiseaseModel, DiseaseState, DiseaseTransition, SISDiseaseModel
from .intervention import TreatmentIntervention
from .mortality import Mortality
from .observer import Observer
from .population import BasePopulation
from .risk import Risk, RiskEffect


def get_model_specification_path():
    from pathlib import Path

    p = Path(__file__).parent / "disease_model.yaml"
    return str(p)


def get_disease_model_simulation():
    from vivarium import InteractiveContext

    p = get_model_specification_path()
    return InteractiveContext(p)
