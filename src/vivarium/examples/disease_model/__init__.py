from .population import BasePopulation
from .mortality import Mortality
from .observer import Observer
from .disease import DiseaseModel, DiseaseState, DiseaseTransition, SIS_DiseaseModel
from .risk import Risk, DirectEffect
from .intervention import MagicWandIntervention


def get_model_specification_path():
    from pathlib import Path
    p = Path(__file__).parent / 'disease_model.yaml'
    return str(p)


def get_disease_model_simulation():
    from vivarium import InteractiveContext
    p = get_model_specification_path()
    return InteractiveContext(p)
