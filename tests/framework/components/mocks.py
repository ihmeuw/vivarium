class MockComponentA:
    def __init__(self, *args):
        self.args = args
        self.builder_used_for_setup = None


class MockComponentB(MockComponentA):
    def setup(self, builder):
        self.builder_used_for_setup = builder

        if len(self.args) > 1:
            children = []
            for arg in self.args:
                children.append(MockComponentB(arg))
            builder.components.add_components(children)
