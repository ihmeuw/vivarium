# mypy: ignore-errors
"""
===========================
Vivarium Command Line Tools
===========================

``vivarium`` provides the tool :command:`simulate` for running simulations
from the command line.  It provides three subcommands:

.. list-table:: ``simulate`` sub-commands
    :header-rows: 1
    :widths: 30, 40

    *   - Name
        - Description
    *   - | **run**
        - | Runs a single simulation from a model specification file.
    *   - | **test**
        - | Runs an example simulation that comes packaged with ``vivarium``.
          | Useful as an installation test.
    *   - | **profile**
        - | Produces a profile of a simulation using the python
          | :mod:`cProfile` module

For more information, see the :ref:`tutorial <cli_tutorial>` on running
simulations from the command line.

.. click:: vivarium.interface.cli:simulate
   :prog: simulate
   :show-nested:

"""

import cProfile
import os
import pstats
from pathlib import Path
from time import time

import click
import yaml
from loguru import logger

from vivarium.examples import disease_model
from vivarium.framework.engine import SimulationContext
from vivarium.framework.logging import (
    configure_logging_to_file,
    configure_logging_to_terminal,
)
from vivarium.framework.utilities import handle_exceptions
from vivarium.interface.utilities import get_output_root


@click.group()
def simulate():
    """A command line utility for running a single simulation.

    You may initiate a new run with the ``run`` sub-command, initiate a test
    run of a provided model specification with the ``test`` subcommand, or
    profile a simulation run with the ``profile`` subcommand.
    """
    pass


@simulate.command()
@click.argument(
    "model_specification", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    "--artifact_path",
    "-i",
    type=click.Path(resolve_path=True),
    help="The path to the artifact data file.",
)
@click.option(
    "--results_directory",
    "-o",
    type=click.Path(resolve_path=True),
    default=Path("~/vivarium_results/").expanduser(),
    help="The directory to write results to. A folder will be created "
    "in this directory with the same name as the configuration file.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Logs verbosely. Useful for debugging and development.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppresses all logging except for warnings and errors.",
)
@click.option(
    "--pdb",
    "with_debugger",
    is_flag=True,
    help="Drop into python debugger if an error occurs.",
)
def run(
    model_specification: Path,
    artifact_path: Path,
    results_directory: Path,
    verbose: bool,
    quiet: bool,
    with_debugger: bool,
):
    """Run a simulation from the command line.

    The simulation itself is defined by the given MODEL_SPECIFICATION yaml file.

    Within the results directory, which defaults to ~/vivarium_results if none
    is provided, a subdirectory will be created with the same name as the
    MODEL_SPECIFICATION if one does not exist. Results will be written to a
    further subdirectory named after the start time of the simulation run.
    """
    if verbose and quiet:
        raise click.UsageError("Cannot be both verbose and quiet.")
    verbosity = 1 + int(verbose) - int(quiet)
    configure_logging_to_terminal(verbosity=verbosity, long_format=False)

    start = time()

    results_root = get_output_root(results_directory, model_specification, artifact_path)
    # Update permissions mask (assign to variable to avoid printing previous value)
    _ = os.umask(0o002)
    results_root.mkdir(parents=True, exist_ok=False)
    output_data_root = results_root / "results"
    output_data_root.mkdir(parents=True, exist_ok=False)

    configure_logging_to_file(output_directory=results_root)

    output_data = {"results_directory": str(output_data_root)}
    input_data = {}
    if artifact_path:
        input_data["artifact_path"] = artifact_path
    override_configuration = {"output_data": output_data, "input_data": input_data}

    sim = SimulationContext(
        model_specification=model_specification, configuration=override_configuration
    )
    with open(f"{results_directory}/model_specification.yaml", "w") as f:
        yaml.dump(sim.model_specification.to_dict(), f)

    main = handle_exceptions(sim.run_simulation, logger, with_debugger)
    main()

    # Save out simulation metadata
    metadata = {}
    metadata["random_seed"] = sim.configuration.randomness.random_seed
    metadata["input_draw"] = sim.configuration.input_data.input_draw_number
    metadata["simulation_run_time"] = time() - start
    metadata["artifact_path"] = artifact_path
    with open(results_root / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    logger.info(f"Simulation finished.\nResults written to {str(output_data_root)}")


@simulate.command()
def test():
    """Run a test simulation using the ``disease_model.yaml`` model specification
    provided in the examples directory.
    """
    configure_logging_to_terminal(verbosity=2, long_format=False)
    model_specification = disease_model.get_model_specification_path()

    sim = SimulationContext(model_specification)

    main = handle_exceptions(sim.run_simulation, logger, with_debugger=False)
    main()

    click.echo()
    click.secho("Installation test successful!", fg="green")


@simulate.command()
@click.argument(
    "model_specification",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--results_directory",
    "-o",
    type=click.Path(resolve_path=True),
    default=Path("~/vivarium_results/").expanduser(),
    help=(
        "The directory to write results to. A folder will be created "
        "in this directory with the same name as the configuration file."
    ),
)
@click.option(
    "--process/--no-process",
    default=False,
    help=(
        "Automatically process the profile to human readable format  with pstats, "
        "sorted by cumulative runtime, and dump to a file"
    ),
)
def profile(model_specification, results_directory, process):
    """Run a simulation based on the provided MODEL_SPECIFICATION and profile the run."""
    model_specification = Path(model_specification)
    results_directory = Path(results_directory)

    out_stats_file = results_directory / f"{model_specification.name}".replace(
        "yaml", "stats"
    )
    sim = SimulationContext(model_specification)
    command = f"sim.run_simulation()"
    cProfile.runctx(command, globals=globals(), locals=locals(), filename=str(out_stats_file))

    if process:
        out_txt_file = results_directory / (out_stats_file.name + ".txt")
        with out_txt_file.open("w") as f:
            p = pstats.Stats(str(out_stats_file), stream=f)
            p.sort_stats("cumulative")
            p.print_stats()
