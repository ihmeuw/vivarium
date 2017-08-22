
class Placeholder:
    def __init__(self, entity_path):
        self.entity_path = entity_path

from ceam_inputs.gbd_mapping import risk_factors, causes
import ceam_inputs as inputs

from ceam_public_health.risks import get_distribution
def DataContainer(entity_path):
    # TODO: Would this be cleaner as a metaclass? Does the fact I'm asking that mean this doesn't make sense?
    entity_type, entity_name = entity_path.split('.')
    if entity_type == 'risk_factor':
        return RiskDataContainer(risk_factors[entity_name])
    if entity_type == 'cause':
        return CauseDataContainer(causess[entity_name])
    if entity_type == 'auxiliary':
        return AuxiliaryDataContainer(entity_name)
    else:
        raise ValueError('Unknown entity type: {}'.format(entity_type))

class _DataContainer:
    def __init__(self, entity):
        self.entity = entity
        self.name = entity if isinstance(entity, str) else entity.name
        self.type = None

class RiskDataContainer(_DataContainer):
    def __init__(self, entity):
        super(RiskDataContainer, self).__init__(entity)
        self.type = 'risk_factor'
        self.tmred = entity.tmred
        self.scale = entity.scale
        self.affected_causes = entity.affected_causes
        self.distribution = entity.distribution

    def get_distribution(self):
        return get_distribution(self.entity)

    def exposure_means(self):
        return inputs.get_exposure_means(risk=self.entity)

    def pafs(self, cause):
        return inputs.get_pafs(risk=self.entity, cause=cause)

    def relative_risks(self, cause):
        return inputs.get_relative_risks(risk=self.entity, cause=cause)

    def mediation_factors(self, cause):
        return inputs.get_mediation_factors(risk=self.entity, cause=cause)

class AuxiliaryDataContainer(_DataContainer):
    def __init__(self, entity):
        super(AuxiliaryDataContainer, self).__init__(entity)
        self.type = 'auxiliary'
        self.data_function = {
                'risk_factor_exposure_correlation_matrices': inputs.load_risk_correlation_matrices
        }[self.entity]

    def data(self):
        return self.data_function()

class CauseDataContainer(_DataContainer):
    def __init__(self, entity):
        super(CauseDataContainer, self).__init__(entity)
        self.type = 'cause'
