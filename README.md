# CEAM (Cost Effectiveness Analysis Microsimulation)

## Installation
1. Clone the repo: `git clone https://#YOUR_STASH_USERNAME#@stash.ihme.washington.edu/scm/cste/ceam.git`
2. Switch to the virtualenv you'll be developing in, if any. #TODO: describe the anaconda workflow?
3. From inside the repo install in development mode: `python setup.py develop`
4. See if it works: `python examples/hello_world.py`
5. If you seem to be missing some dependencies, try installing them with pip: `pip install -r requirements.txt` #TODO: or conda?

## Path to the Data
By default CEAM assumes that your filesystem is like the one on the cluster. If that's not the case then you'll need to tell the system where to find the datafiles. Do this by creating a file called `ceam.cfg` in your home directory. It's contents should look like this:

    [general]
    reference_data_directory=PATH_TO_THE_J_DRIVE/Project/Cost_Effectiveness/dev/data_processed
    population_data_directory=PATH_TO_THE_J_DRIVE/Project/Cost_Effectiveness/dev/data_processed/population_columns

Protip: You can override any configuration using this file. Take a look at ceam/config.cfg to get an idea what other configuration variables there are.

## Testing
All tests are in the ceam_tests directory. Test files should correspond with the files they test. So, `ceam/engine.py` will have `ceam_tests/test_engine.py`. Run the tests by invoking `py.test`. Some of the data backed tests are slow, if that's irritating you can exclude them with `py.test -m "not slow"`.

## Development Process
Our basic development process will use a `master` branch for major releases (corresponding to presentations or papers), a `develop` branch which has the current shared version of the code and should always be as stable and bug free as possible, and many feature branches which have work-in-progress code for new features.

### How to develop a new feature
1. Assure that you have a current version of the repository:
    git pull
2. Create your feature branch. Branch names should be reasonably descriptive. `salt_consumption_model` not `stuff`:
    ```
    git checkout -b YOUR_BRANCH_NAME develop
    ```
    * Alternately, if your work is tied to a JIRA ticket you can create the branch using the 'Create Branch' link from inside the ticket and then checkout your new branch like so:
        ```
        git checkout --track -b origin/YOUR_BRANCH_NAME
        ```
3. Write some fancy code.
4. Commit frequently (remember you and your collaborators are the only ones who are looking at this code, so it's better to have a fine grained record of your work than to make sure everything is perfect before you commit):
    ```
    git add PATH_TO_CHANGED_FILE
    git commit -m "Write a short but thorough description of the changes. If that's hard, you should probably break it up into multiple commits."
    git push
    ```
5. Repeat steps 3-4 until everything works nicely (don't forget to write tests to prove that things really do work (you will find bugs while writing the tests, I guarantee it)).
6. Create a pull request: https://stash.ihme.washington.edu/projects/CSTE/repos/ceam/pull-requests?create
7. Once all your reviewers agree that things are good, use the pull request interface to merge your feature branch back into `develop`
8. :partyhat:

## Run it on the cluster
To run a test on the cluster use 'scripts/cluster_runner.py' This will launch and optionally monitor any number of simulation runs distributed across the cluster. Example usage:
    ```
    cluster_runner.py --runs 300 --runs_per_slot 10 --results_dir ~/ --watch_progress
    ```

This will create a directory in your home directory named for the current time and write all results files there. It will launch 30 jobs running the simulation 10 times each. You'll end up with 30 result csv files, one for each process. The cluster_runner script will watch the jobs until they finish, occasionally printing out progress information.
