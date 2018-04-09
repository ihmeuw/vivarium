import argparse
from bdb import BdbQuit
from vivarium.framework.engine import do_command

import logging
_log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'list_datasets'])
    parser.add_argument('simulation_configuration', nargs='?', default=None, type=str)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--random_seed', '-s', type=int, default=0, help="Seed for random number generation")
    parser.add_argument('--results_path', '-o', type=str, default=None, help='Output directory to write results to')
    parser.add_argument('--log', type=str, default=None, help='Path to log file')
    parser.add_argument('--pdb', action='store_true', help='Run in the debugger')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.ERROR
    logging.basicConfig(filename=args.log, level=log_level)

    try:
        do_command(args)
    except (BdbQuit, KeyboardInterrupt):
        raise
    except Exception as e:
        if args.pdb:
            import pdb
            import traceback
            traceback.print_exc()
            pdb.post_mortem()
        else:
            logging.exception("Uncaught exception {}".format(e))
            raise
