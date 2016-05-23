# ~/ceam/examples/hello_world.py

from datetime import datetime, timedelta

from engine import Simulation
from modules.ihd import IHDModule
from modules.metrics import MetricsModule


def main():
    simulation = Simulation()

    module = IHDModule()
    module.setup()
    simulation.register_modules([module])

    metrics_module = MetricsModule()
    metrics_module.setup()
    simulation.register_modules([metrics_module])

    simulation.load_population('/home/j/Project/Cost_Effectiveness/dev/data_processed/population_columns')
    simulation.load_data('/home/j/Project/Cost_Effectiveness/dev/data_processed')

    simulation.run(datetime(1990, 1, 1), datetime(2013, 12, 31), timedelta(days=30.5)) #TODO: Is 30.5 days a good enough approximation of one month? -Alec

    print(metrics_module.metrics)


if __name__ == '__main__':
    main()


# End.
