/*This file uses jenkins shared library `vivarium_build_utils`,
found at https://github.com/ihmeuw/vivarium_build_utils
Due to Jenkins convention, importable modules must be stored
in the 'vars' folder.
Jenkins needs to be configured globally to use the correct branch.
To configure the repo/branch go to:
* Manage Jenkins
  * Configure System
    * Global Pipeline Libraries section
      * Library subsection
        * Name: The Name for the lib
        * Version: The branch you want to use. Throws an error
                   for nonexistent branches.
        * Project Repository: Url to the shared lib
        * Credentials: SSH key to access the repo

Updating the shared repo will take affect on the next pipeline invocation.
The "_" denotes that all modules will be imported from the shared library.
*/ 
@Library("vivarium_build_utils") _
reusable_pipeline(scheduled_branches: ["main"], upstream_repos: ["layered_config_tree"])
