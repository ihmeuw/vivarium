# ~/ceam/examples/hello_world.py

from datetime import datetime, timedelta

from ceam.engine import Simulation
from ceam.modules.ihd import IHDModule
from ceam.modules.blood_pressure import BloodPressureModule
from ceam.modules.metrics import MetricsModule


def main():
    simulation = Simulation()

    module = BloodPressureModule()
    module.setup()
    simulation.register_modules([module])

    module = IHDModule()
    module.setup()
    simulation.register_modules([module])

    metrics_module = MetricsModule()
    metrics_module.setup()
    simulation.register_modules([metrics_module])

    simulation.load_population()
    simulation.load_data()

    simulation.run(datetime(1990, 1, 1), datetime(2013, 12, 31), timedelta(days=30.5)) #TODO: Is 30.5 days a good enough approximation of one month? -Alec

    print(metrics_module.metrics)


if __name__ == '__main__':
    main()


# End.
