import math
from itertools import product
from pathlib import Path
from time import time

import dill
import pandas as pd
import pytest

from tests.framework.results.helpers import (
    FAMILIARS,
    HARRY_POTTER_CONFIG,
    POWER_LEVEL_GROUP_LABELS,
    STUDENT_HOUSES,
    Hogwarts,
    HogwartsResultsStratifier,
    HousePointsObserver,
    NoStratificationsQuidditchWinsObserver,
    QuidditchWinsObserver,
)
from tests.helpers import Listener, MockComponentA, MockComponentB
from vivarium import Component
from vivarium.framework.artifact import ArtifactInterface, ArtifactManager
from vivarium.framework.components import (
    ComponentConfigError,
    ComponentInterface,
    ComponentManager,
    OrderedComponentSet,
)
from vivarium.framework.engine import Builder
from vivarium.framework.engine import SimulationContext as SimulationContext_
from vivarium.framework.event import EventInterface, EventManager
from vivarium.framework.lifecycle import LifeCycleInterface, LifeCycleManager
from vivarium.framework.logging import LoggingInterface, LoggingManager
from vivarium.framework.lookup import LookupTableInterface, LookupTableManager
from vivarium.framework.population import PopulationInterface, PopulationManager
from vivarium.framework.randomness import RandomnessInterface, RandomnessManager
from vivarium.framework.resource import ResourceInterface, ResourceManager
from vivarium.framework.results import VALUE_COLUMN, ResultsInterface, ResultsManager
from vivarium.framework.time import DateTimeClock, TimeInterface
from vivarium.framework.values import ValuesInterface, ValuesManager


def is_same_object_method(m1, m2):
    return m1.__func__ is m2.__func__ and m1.__self__ is m2.__self__


@pytest.fixture()
def SimulationContext():
    yield SimulationContext_
    SimulationContext_._clear_context_cache()


@pytest.fixture
def components():
    return [
        MockComponentA("gretchen", "whimsy"),
        Listener("listener"),
        MockComponentB("spoon", "antelope", "23"),
    ]


@pytest.fixture
def log(mocker):
    return mocker.patch("vivarium.framework.logging.manager.loguru.logger")


def test_simulation_with_non_components(SimulationContext, components: list[Component]):
    class NonComponent:
        def __init__(self):
            self.name = "non_component"

    with pytest.raises(ComponentConfigError):
        SimulationContext(components=components + [NonComponent()])


def test_SimulationContext_get_sim_name(SimulationContext):
    assert SimulationContext._created_simulation_contexts == set()

    assert SimulationContext._get_context_name(None) == "simulation_1"
    assert SimulationContext._get_context_name("foo") == "foo"

    assert SimulationContext._created_simulation_contexts == {"simulation_1", "foo"}


def test_SimulationContext_init_default(SimulationContext, components):
    sim = SimulationContext(components=components)

    assert isinstance(sim._logging, LoggingManager)
    assert isinstance(sim._lifecycle, LifeCycleManager)
    assert isinstance(sim._component_manager, ComponentManager)
    assert isinstance(sim._clock, DateTimeClock)
    assert isinstance(sim._values, ValuesManager)
    assert isinstance(sim._events, EventManager)
    assert isinstance(sim._population, PopulationManager)
    assert isinstance(sim._resource, ResourceManager)
    assert isinstance(sim._results, ResultsManager)
    assert isinstance(sim._tables, LookupTableManager)
    assert isinstance(sim._randomness, RandomnessManager)
    assert isinstance(sim._data, ArtifactManager)

    assert isinstance(sim._builder, Builder)
    assert sim._builder.configuration is sim.configuration

    assert isinstance(sim._builder.logging, LoggingInterface)
    assert sim._builder.logging._manager is sim._logging
    assert isinstance(sim._builder.lookup, LookupTableInterface)
    assert sim._builder.lookup._manager is sim._tables
    assert isinstance(sim._builder.value, ValuesInterface)
    assert sim._builder.value._manager is sim._values
    assert isinstance(sim._builder.event, EventInterface)
    assert sim._builder.event._manager is sim._events
    assert isinstance(sim._builder.population, PopulationInterface)
    assert sim._builder.population._manager is sim._population
    assert isinstance(sim._builder.resources, ResourceInterface)
    assert sim._builder.resources._manager is sim._resource
    assert isinstance(sim._builder.results, ResultsInterface)
    assert sim._builder.results._manager is sim._results
    assert isinstance(sim._builder.randomness, RandomnessInterface)
    assert sim._builder.randomness._manager is sim._randomness
    assert isinstance(sim._builder.time, TimeInterface)
    assert sim._builder.time._manager is sim._clock
    assert isinstance(sim._builder.components, ComponentInterface)
    assert sim._builder.components._manager is sim._component_manager
    assert isinstance(sim._builder.lifecycle, LifeCycleInterface)
    assert sim._builder.lifecycle._manager is sim._lifecycle
    assert isinstance(sim._builder.data, ArtifactInterface)
    assert sim._builder.data._manager is sim._data

    # Ordering matters.
    managers = [
        sim._logging,
        sim._lifecycle,
        sim._resource,
        sim._values,
        sim._population,
        sim._clock,
        sim._randomness,
        sim._events,
        sim._tables,
        sim._data,
        sim._results,
    ]
    assert sim._component_manager._managers == OrderedComponentSet(*managers)
    unpacked_components = []
    for c in components:
        unpacked_components.append(c)
        if hasattr(c, "sub_components"):
            unpacked_components.extend(c.sub_components)
    assert list(sim._component_manager._components) == unpacked_components


def test_SimulationContext_name_management(SimulationContext):
    assert SimulationContext._created_simulation_contexts == set()

    sim1 = SimulationContext()
    assert sim1._name == "simulation_1"
    assert SimulationContext._created_simulation_contexts == {"simulation_1"}

    sim2 = SimulationContext(sim_name="foo")
    assert sim2._name == "foo"
    assert SimulationContext._created_simulation_contexts == {"simulation_1", "foo"}

    sim3 = SimulationContext()
    assert sim3._name == "simulation_3"
    assert SimulationContext._created_simulation_contexts == {
        "simulation_1",
        "foo",
        "simulation_3",
    }


def test_SimulationContext_run_simulation(SimulationContext, mocker):
    sim = SimulationContext()

    expected_calls = [
        "setup",
        "initialize_simulants",
        "run",
        "finalize",
        "report",
    ]

    # Mock the methods called by sim.run()
    mock = mocker.MagicMock()
    for call in expected_calls:
        mock.attach_mock(
            mocker.patch(f"vivarium.framework.engine.SimulationContext.{call}"), call
        )

    sim.run_simulation()

    # Assert the calls are each made exactly once and in the correct order
    # NOTE: mock.mock_calls is a list like [call.setup(), call.initialize_simulants(), ...]
    actual_calls = [str(call).split("call.")[1].split("()")[0] for call in mock.mock_calls]
    assert actual_calls == expected_calls


def test_SimulationContext_setup_default(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if "listener" in c.args][0]
    assert not listener.post_setup_called
    sim.setup()

    unpacked_components = []
    for c in components:
        unpacked_components.append(c)
        if hasattr(c, "sub_components"):
            unpacked_components.extend(c.sub_components)

    for a, b in zip(sim._component_manager._components, unpacked_components):
        assert type(a) == type(b)
        if hasattr(a, "args"):
            assert a.args == b.args

    assert is_same_object_method(sim.simulant_creator, sim._population._create_simulants)
    assert sim.time_step_events == [
        "time_step__prepare",
        "time_step",
        "time_step__cleanup",
        "collect_metrics",
    ]
    for k in sim.time_step_emitters.keys():
        assert is_same_object_method(
            sim.time_step_emitters[k], sim._events._event_types[k].emit
        )

    assert is_same_object_method(
        sim.end_emitter, sim._events._event_types["simulation_end"].emit
    )

    assert listener.post_setup_called


def test_SimulationContext_initialize_simulants(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    pop_size = sim.configuration.population.population_size
    current_time = sim._clock.time
    assert sim._population.get_population(True).empty
    sim.initialize_simulants()
    pop = sim._population.get_population(True)
    assert len(pop) == pop_size
    assert sim._clock.time == current_time


def test_SimulationContext_step(SimulationContext, log, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()

    current_time = sim._clock.time
    step_size = sim._clock.step_size

    listener = [c for c in components if "listener" in c.args][0]

    assert not listener.time_step_prepare_called
    assert not listener.time_step_called
    assert not listener.time_step_cleanup_called
    assert not listener.collect_metrics_called

    sim.step()

    assert log.debug.called_once_with(current_time)
    assert listener.time_step_prepare_called
    assert listener.time_step_called
    assert listener.time_step_cleanup_called
    assert listener.collect_metrics_called

    assert sim._clock.time == current_time + step_size


def test_SimulationContext_finalize(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if "listener" in c.args][0]
    sim.setup()
    sim.initialize_simulants()
    sim.step()
    assert not listener.simulation_end_called
    sim.finalize()
    assert listener.simulation_end_called


def test_get_results(SimulationContext, base_config):
    """Test that get_results returns expected values. This does NOT test for
    correct formatting.
    """
    components = [
        Hogwarts(),
        HousePointsObserver(),
        NoStratificationsQuidditchWinsObserver(),
        QuidditchWinsObserver(),
        HogwartsResultsStratifier(),
    ]
    sim = SimulationContext(base_config, components, configuration=HARRY_POTTER_CONFIG)
    sim.run_simulation()
    for measure, results in sim.get_results().items():
        raw_results = sim._results._raw_results[measure].sort_index()
        assert results.set_index(raw_results.index.names)[[VALUE_COLUMN]].equals(raw_results)


def test_SimulationContext_report_no_write_warning(SimulationContext, base_config, caplog):
    components = [
        Hogwarts(),
        HousePointsObserver(),
        NoStratificationsQuidditchWinsObserver(),
        QuidditchWinsObserver(),
        HogwartsResultsStratifier(),
    ]
    sim = SimulationContext(base_config, components, configuration=HARRY_POTTER_CONFIG)
    sim.run_simulation()
    assert "No results directory set; results are not written to disk." in caplog.text
    results = sim.get_results()
    assert set(results) == set(
        ["house_points", "quidditch_wins", "no_stratifications_quidditch_wins"]
    )
    assert all([isinstance(df, pd.DataFrame) for df in results.values()])


def test_SimulationContext_report_write(SimulationContext, base_config, components, tmpdir):
    """Test that the written results match get_results"""
    results_root = Path(tmpdir)
    configuration = {"output_data": {"results_directory": str(results_root)}}
    configuration.update(HARRY_POTTER_CONFIG)
    components = [
        Hogwarts(),
        HousePointsObserver(),
        NoStratificationsQuidditchWinsObserver(),
        QuidditchWinsObserver(),
        HogwartsResultsStratifier(),
    ]
    sim = SimulationContext(base_config, components, configuration)
    sim.run_simulation()
    # Check for expected results written
    results_list = [file for file in results_root.rglob("*.parquet")]
    assert set([file.name for file in results_list]) == set(
        [
            "house_points.parquet",
            "quidditch_wins.parquet",
            "no_stratifications_quidditch_wins.parquet",
        ]
    )

    # Check that written results match get_results method
    for measure in ["house_points", "quidditch_wins", "no_stratifications_quidditch_wins"]:
        results = sim.get_results()[measure]
        written_results = pd.read_parquet(results_root / f"{measure}.parquet")
        assert results.equals(written_results)


def test_SimulationContext_write_backup(mocker, SimulationContext, tmpdir):
    # TODO MIC-5216: Remove mocks when we can use dill in pytest.
    mocker.patch("vivarium.framework.engine.dill.dump")
    mocker.patch("vivarium.framework.engine.dill.load", return_value=SimulationContext())
    sim = SimulationContext()
    backup_path = tmpdir / "backup.pkl"
    sim.write_backup(backup_path)
    assert backup_path.exists()
    with open(backup_path, "rb") as f:
        sim_backup = dill.load(f)
    assert isinstance(sim_backup, SimulationContext)


def test_SimulationContext_run_with_backup(mocker, SimulationContext, base_config, tmpdir):
    mocker.patch("vivarium.framework.engine.SimulationContext.write_backup")
    original_time = time()

    def time_generator():
        current_time = original_time
        while True:
            yield current_time
            current_time += 5

    mocker.patch("vivarium.framework.engine.time", side_effect=time_generator())
    components = [
        Hogwarts(),
        HousePointsObserver(),
        NoStratificationsQuidditchWinsObserver(),
        QuidditchWinsObserver(),
        HogwartsResultsStratifier(),
    ]
    sim = SimulationContext(base_config, components, configuration=HARRY_POTTER_CONFIG)
    backup_path = tmpdir / "backup.pkl"
    sim.setup()
    sim.initialize_simulants()
    sim.run(backup_path=backup_path, backup_freq=5)
    assert sim.write_backup.call_count == _get_num_steps(sim)


def test_get_results_formatting(SimulationContext, base_config):
    """Test formatted results are as expected"""
    components = [
        Hogwarts(),
        HousePointsObserver(),
        NoStratificationsQuidditchWinsObserver(),
        QuidditchWinsObserver(),
        HogwartsResultsStratifier(),
    ]
    sim = SimulationContext(base_config, components, configuration=HARRY_POTTER_CONFIG)
    sim.run_simulation()
    num_steps = _get_num_steps(sim)

    house_points = sim.get_results()["house_points"]
    quidditch_wins = sim.get_results()["quidditch_wins"]
    no_stratifications_quidditch_wins = sim.get_results()["no_stratifications_quidditch_wins"]

    # Check that each dataset includes the entire cartesian product of stratifications
    # (or, when no stratifications, just a single "all" row)
    assert set(zip(house_points["student_house"], house_points["power_level_group"])) == set(
        product(STUDENT_HOUSES, POWER_LEVEL_GROUP_LABELS)
    )
    assert set(quidditch_wins["familiar"]) == set(FAMILIARS)
    assert no_stratifications_quidditch_wins.shape[0] == 1
    assert (no_stratifications_quidditch_wins["stratification"] == "all").all()
    assert set(quidditch_wins.columns).difference(
        set(no_stratifications_quidditch_wins.columns)
    ) == set(["familiar"])

    # Set up filters for groups that scored points
    house_points_filter = (house_points["student_house"] == "gryffindor") & (
        house_points["power_level_group"].isin(["low", "very high"])
    )
    quidditch_wins_filter = quidditch_wins["familiar"] == "banana_slug"
    no_strats_quidditch_wins_filter = (
        no_stratifications_quidditch_wins["stratification"] == "all"
    )
    for measure, filter in [
        ("house_points", house_points_filter),
        ("quidditch_wins", quidditch_wins_filter),
        ("no_stratifications_quidditch_wins", no_strats_quidditch_wins_filter),
    ]:
        # Check columns
        df = eval(measure)
        # Check that metrics col matches name of dataset
        assert (df["measure"] == measure).all()
        # We do enforce a col order, but most importantly ensure VALUE_COLUMN is at the end
        assert df.columns[-1] == VALUE_COLUMN
        # Check values
        # Check that all values are 0 except for expected groups
        assert (df.loc[filter, VALUE_COLUMN] != 0).all()
        assert (df.loc[~filter, VALUE_COLUMN] == 0).all()
        # Check that expected groups' values are a multiple of the number of steps
        assert (df.loc[filter, VALUE_COLUMN] % num_steps == 0).all()


####################
# HELPER FUNCTIONS #
####################
def _convert_to_datetime(date_dict: dict[str, int]) -> pd.Timestamp:
    return pd.to_datetime(
        "-".join([str(val) for val in date_dict.values()]), format="%Y-%m-%d"
    )


def _get_num_steps(sim: SimulationContext) -> int:
    time_dict = sim.configuration.time.to_dict()
    end_date = _convert_to_datetime(time_dict["end"])
    start_date = _convert_to_datetime(time_dict["start"])
    num_steps = math.ceil((end_date - start_date).days / time_dict["step_size"])
    return num_steps
