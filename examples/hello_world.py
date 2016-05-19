from engine import Simulation
from modules.ihd import IHDModule

def main():
    simulation = Simulation()

    ihd_module = IHDModule()
    ihd_module.setup()
    simulation.register_module(ihd_module)

    simulation.load_population('/home/j/Project/Cost_Effectiveness/dev/data_processed/population_columns')
    simulation.load_data('/home/j/Project/Cost_Effectiveness/dev/data_processed')

    simulation.run(1990, 2013)

    print('YLDs: %s'%sum(simulation.yld_by_year.values()))

if __name__ == '__main__':
    main()
