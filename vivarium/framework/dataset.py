class Placeholder:
    def __init__(self, entity_path):
        self.entity_path = entity_path

class DatasetManager:
    def construct_data_container(self, entity_path):
        raise NotImplementedError()
